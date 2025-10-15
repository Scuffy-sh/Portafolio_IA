'''Clasificación de intents con embeddings (Sentence-BERT)

Reconocimiento de entidades (NER con spaCy)

Pipeline completo NLP → Modelo → Respuesta

Entorno recomendado: "chatbot"'''


# Importar librerías necesarias
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import spacy
import re
import random
from datetime import datetime

# Inicialización de la app
st.title("Chatbot de Reservas")

# Definir función para cargar el modelo de embeddings
@st.cache_resource
def load_embedding_model():
    # Usamos un modelo ligero para embeddings
    return SentenceTransformer('all-MiniLM-L6-v2')

# Cargar el modelo de embeddings
embed_model = load_embedding_model()


# Definir función para cargar el modelo NER de spaCy
@st.cache_resource
def load_ner_model():
    return spacy.load("es_core_news_sm") # Modelo en español

# Cargar el modelo NER
ner_model = load_ner_model()

# Definir intents y ejemplos
intents = {
    "saludo": ["hola", "buenas", "buenas tardes", "buenos días"],
    "despedida": ["adios", "hasta luego", "nos vemos"],
    "reservar_mesa": [
        "Quiero reservar una mesa",
        "Reservar 2 personas para mañana",
        "Necesito una mesa para 4 personas a las 20:00"
    ],
    "cancelar_reserva": ["Quiero cancelar mi reserva", "Cancelar mesa", "No podré ir"],
    "pregunta_menu": ["Qué menú tienen", "Cuál es el menú del día", "Quiero ver el menú"]
}

# Precalcular embeddings de ejemplos
examples_embeddings = {}
# Calcular embeddings para cada intent
for intent, examples in intents.items():
    # Calcular embeddings de los ejemplos
    embeddings = embed_model.encode(examples, convert_to_tensor=True)
    # Guardar en un diccionario
    examples_embeddings[intent] = embeddings
    

# Historial de la conversación
if 'history' not in st.session_state:
    st.session_state.history = []
    

# Función de clasificación de intents
def predict_intent(user_input):
    # Calcular embedding del input del usuario
    input_embedding = embed_model.encode(user_input, convert_to_tensor=True)
    max_sim = -1
    best_intent = None
    
    # Comparar con cada intent
    for intent, embeddings in examples_embeddings.items():
        sim_scores = util.cos_sim(input_embedding, embeddings)
        sim_score = sim_scores.max().item()
        if sim_score > max_sim:
            max_sim = sim_score
            best_intent = intent
    return best_intent, max_sim

# Función de extracción de entidades
def extract_entities(user_input):
    doc = ner_model(user_input)
    entities = {}
    # Extraer entidades reconocidas
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities

# Generación de respuesta
def generate_response(intent, entities):
    if intent == "saludo":
        return random.choice(["¡Hola! ¿Deseas reservar una mesa?", "¡Buenas! ¿En qué puedo ayudarte?"])
    elif intent == "despedida":
        return random.choice(["¡Hasta luego!", "Que tengas un buen día."])
    elif intent == "reservar_mesa":
        num_personas = entities.get("CARDINAL", "1")
        fecha = entities.get("DATE", "la fecha deseada")
        hora = entities.get("TIME", "la hora que prefieras")
        return f"He reservado una mesa para {num_personas} personas el {fecha} a las {hora}."
    elif intent == "cancelar_reserva":
        return "Tu reserva ha sido cancelada."
    elif intent == "pregunta_menu":
        return "Nuestro menú incluye pasta, ensaladas y platos del día."
    else:
        return "Lo siento, no he entendido tu solicitud."
    
# Interfaz de usuario
chat_placeholder = st.container()
user_input = st.text_input("Escribe tu mensaje aquí:")

if user_input:
    # Predecir intent
    intent, sim = predict_intent(user_input)
    
    # Extraer entidades
    entities = extract_entities(user_input)
    
    # Generar respuesta
    bot_response = generate_response(intent, entities)
    
    # Actualizar el historial de la conversación
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "bot", "content": bot_response})
    
# Mostrar el historial de la conversación
with chat_placeholder :
    for message in st.session_state.history[-10:]: # Mostrar solo los últimos 10 mensajes
        if message['role'] == 'user':   
            st.markdown(f"Tú: {message['content']}")
        else:
            st.markdown(f"Chatbot: {message['content']}")
        st.markdown("-" * 40)