from langchain_openai import ChatOpenAI
import os
from flask import Flask, jsonify, request
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool


## datos de trazabilidad
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt..."
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "amcagent"
os.environ["OPENAI_API_KEY"] ="sk-proj..."


app = Flask(__name__)

@app.route('/agent', methods=['GET'])
def main():
    #Capturamos variables enviadas
    id_agente = request.args.get('idagente')
    msg = request.args.get('msg')
    #datos de configuracion
    DB_URI = os.environ.get(
        "DB_URI",
        "postgresql://usuario:password@0.0.0.0:5432/basededatos?sslmode=disable"
    )
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    DB_URI_SQLALCHEMY = DB_URI.replace("postgresql://", "postgresql+psycopg://", 1)
    db = SQLDatabase.from_uri(DB_URI_SQLALCHEMY)

    def get_schema() -> str:
        return db.get_table_info()

    prompt_sql = ChatPromptTemplate.from_template("""
    Dado el siguiente esquema de la base de datos y una pregunta, genera solamente una consulta SQL válida y completa.

    Si la pregunta menciona un código de pedido, genera una consulta que:
    Incluya el estado del pedido y la fecha de entrega.
    Traiga los productos relacionados a ese pedido.
    No uses comillas triples, bloques de código ni explicaciones.

    Esquema de la base de datos:
    {schema}

    Pregunta del usuario:
    {question}

    Consulta SQL (sin formato adicional):
    """)

    sql_chain = prompt_sql | ChatOpenAI(model="gpt-4.1-2025-04-14").bind(stop=["\nSQLResult"]) | StrOutputParser()

    def generar_sql(schema: str, question: str) -> str:
        return sql_chain.invoke({"schema": schema, "question": question})

    def run_query(query: str) -> str:
        return db.run(query)

    prompt_respuesta = ChatPromptTemplate.from_template("""
    Basado en el esquema, pregunta, SQL y resultado, responde en lenguaje natural.

    Esquema:
    {schema}

    Pregunta:
    {question}

    SQL:
    {query}

    Resultado:
    {response}

    Respuesta natural:
    """)

    sqlnatural_chain = prompt_respuesta | ChatOpenAI(model="gpt-4.1-2025-04-14") | StrOutputParser()

    def generar_respuesta(schema: str, question: str, query: str, response: str) -> str:
        return sqlnatural_chain.invoke({
            "schema": schema,
            "question": question,
            "query": query,
            "response": response
        })
    
    def consultar_pedido(pregunta: str) -> str:
        schema = get_schema()
        sql = generar_sql(schema, pregunta)
        resultado = run_query(sql)
        return generar_respuesta(schema, pregunta, sql, resultado)
    
    tool_consulta_pedido = Tool(
        name="consulta_pedido",
        func=consultar_pedido,
        description="Usa esta herramienta para obtener informacion detallada de un pedido."
    )
    db_vector = ElasticsearchStore(
        es_url="http://0.0.0.0:9200",
        es_user="elastic",
        es_password="clave",
        index_name="lg-politicas",
        embedding=OpenAIEmbeddings())

    retriever = db_vector.as_retriever()

    tool_rag =retriever.as_tool(
        name="politicas",
        description="Consulta solo politicas de reclamos y reembolsos para determinar si procede o no el reembolso en un caso específico.",
    )
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Eres un asistente experto en seguimiento de pedidos y reclamos de una tienda de computadoras. Atiendes a los clientes de forma clara, empática y breve, como un amigo experto en tecnología.

    FLUJO DE ATENCIÓN:

    1. **Saludo e identificación**:
    - Saluda cordialmente.
    - Si el cliente no da un código de pedido, solicítalo antes de continuar.

    2. **Verificación del pedido**:
    - Si el pedido está en curso, solo indícale que puede consultar el estado en la web.

    3. **Detección del problema**:
    - Muestra al cliente los productos del pedido.
    - Pregunta cuál(es) tienen problemas y qué tipo: **Faltante**, **Maltratado** o **Erróneo**.
    - Solo continúas si el problema es uno de esos tres.

    4. **Revisión y resolución**:
    - Verifica si el producto figura en el pedido.
    - Usa la herramienta para consultar las políticas oficiales.
    - Si tienes la fecha de entrega, calcula tú si el reclamo está dentro del plazo de 3 días hábiles (no preguntes al cliente).
    - Si todo aplica, calcula el monto exacto del reembolso y comunícalo de forma directa:
        - Menciona el porcentaje y el monto final.
        - Indica que el reembolso será en **créditos de la app**, a menos que el cliente prefiera su tarjeta.
    - **No repitas el texto de la política al cliente. Solo aplica la decisión con lenguaje natural.**

    5. **Cierre**:
    - Agradece al cliente y pregunta si necesita algo más.

    REGLAS CLAVE:

    - Usa solo las herramientas disponibles.
    - No inventes reglas, productos ni estados.
    - No confirmes reembolsos si el producto no figura, el problema no es válido o el plazo expiró.
    - **No muestres ni expliques el texto exacto de las políticas al cliente. Aplica el criterio directamente.**
    - El reembolso por defecto se hace en **créditos de la app**. Solo si el cliente los rechaza, se ofrece devolución a tarjeta (5 a 10 días hábiles).
    """),
        ("human", "{messages}"),
    ])

    with ConnectionPool(
            # Example configuration
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs,
    ) as pool:
        checkpointer = PostgresSaver(pool)

        # Inicializamos el modelo
        model = ChatOpenAI(model="gpt-4.1-2025-04-14")

        # Agrupamos las herramientas
        toolkit = [tool_rag, tool_consulta_pedido]

        # inicializamos el agente
        agent_executor = create_react_agent(
            model=model,
            tools=toolkit,
            checkpointer=checkpointer,
            prompt=prompt
        )
        # ejecutamos el agente
        config = {"configurable": {"thread_id": id_agente}}
        response = agent_executor.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        return response['messages'][-1].content


if __name__ == '__main__':
    # La aplicación escucha en el puerto 8080, requerido por Cloud Run
    app.run(host='0.0.0.0', port=8080)