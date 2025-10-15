# Librerías necesarias
import streamlit as st
import nltk, random, os, unicodedata
from nltk.stem import SnowballStemmer

# Configurar la ruta para los datos de NLTK
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')

# Crear el directorio si no existe
os.makedirs(nltk_data_path, exist_ok=True)

# Descargar recursos necesarios de NLTK en la ruta especificada
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

# Añadir la ruta a la configuración de NLTK
nltk.data.path.append(nltk_data_path)

# Importar word_tokenize después de configurar la ruta de datos
from nltk.tokenize import word_tokenize

# Configuración del Stemmer en español
stemmer = SnowballStemmer('spanish')

# Diccionarios de intents y responses
intents = {
    "saludo": ["hola", "buenas", "qué tal"],
    "despedida": ["adios", "chao", "hasta luego"],
    "ayuda": ["ayuda", "duda", "pregunta"],
    "producto": ["teléfono", "móvil", "portátil", "cargador"],
    "horario": ["horario", "abierto", "cierra", "apertura"],
    "envio": ["envío", "entrega", "reparto", "llegada"]
}

responses = {
    "saludo": [ 
        "¡Hola! Bienvenido a ElectroStore, ¿en qué puedo ayudarte?",
        "¡Buenas! ¿En qué puedo ayudarte?",
        "¡Hey! Gracias por visitarnos. ¿Qué buscas hoy?"
    ],
    "despedida": [
        "¡Gracias por visitarnos! Que tengas un buen día",
        "¡Hasta luego! No dudes en volver si tienes más preguntas.",
        "¡Chao! Esperamos verte pronto."
    ],
    "ayuda": [
        "Estoy aquí para ayudarte. Puedes preguntarme por horarios, envíos o productos.",
        "No dudes en preguntar cualquier duda que tengas.",
        "Estoy a tu disposición para resolver tus inquietudes."
    ],
    "horario": [
        "Nuestro horario de atención es de lunes a viernes de 9:00 a 18:00.",
        "Estamos abiertos de lunes a viernes de 9:00 a 18:00.",
        "Puedes visitarnos de lunes a viernes de 9:00 a 18:00."
    ],
    "envio": [
        "El coste de envío depende de la ubicación y el peso del paquete.",
        "Ofrecemos envío gratuito en pedidos superiores a 50€.",
        "Los gastos de envío se calculan al finalizar la compra."
    ],
    "producto": [
        "Todos nuestros productos están disponibles en nuestra tienda online.",
        "Puedes consultar la disponibilidad de un artículo en nuestra web.",
        "Si un producto no está en stock, puedes solicitar que te avisemos cuando vuelva a estar disponible."
    ]
}

# Pesos de los intents
intents_weights = {
    "saludo": 0.5,
    "despedida": 0.5,
    "ayuda": 0.5,
    "horario": 1,
    "envio": 1,
    "producto": 1,
}

# Variable de contexto
if 'context' not in st.session_state:
    st.session_state.context = None

# Historial de conversación
if 'history' not in st.session_state:
    st.session_state.history = []

# Función del chatbot con contexto
def chatbot_context(user_input):
    # Tokenizar y stemizar la entrada del usuario
    tokens = [stemmer.stem(t) for t in word_tokenize(user_input.lower())]

    # Inicializar variables para el mejor intent y la puntuación máxima
    best_intent = None
    max_score = 0
    
    # Evaluar cada intent y calcular su puntuación
    for intent, keywords in intents.items():
        stemmed_keywords = [stemmer.stem(k) for k in keywords]
        matches = sum(1 for token in tokens if token in stemmed_keywords)
        score = matches * intents_weights.get(intent, 1)
        # Actualizar el mejor intent si la puntuación es mayor
        if score > max_score:
            max_score = score
            best_intent = intent
            
    # Si no hay coincidencias, usar el contexto anterior
    if best_intent is None and st.session_state.context is not None:
        best_intent = st.session_state.context
        
    # Actualizar el contexto si se detecta un nuevo intent
    if best_intent is not None and max_score > 0:
        st.session_state.context = best_intent
    
    # Si se encuentra un intent, devolver una respuesta aleatoria 
    if best_intent:
        return random.choice(responses.get(best_intent))
    
    else: 
        return "Lo siento, no he entendido tu pregunta. ¿Puedes reformularla?"

# Función para limpiar el input del usuario
def clear_input():
    st.session_state.user_input = ""

# Interfaz de usuario con Streamlit
st.title("Chatbot FAQ - ElectroStore")

# Entrada del usuario
user_input = st.chat_input("Escribe tu mensaje aquí:")

if user_input:
    # Obtener la respuesta del chatbot
    bot_response = chatbot_context(user_input)
                
    # Actualizar el historial de la conversación
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "bot", "content": bot_response})
    
# Mostrar el historial de la conversación
for message in st.session_state.history:
    if message['role'] == 'user':
        st.markdown(f"<div style='background-color:#DCF8C6; color:black; padding:10px; border-radius:10px; width:fit-content; margin-bottom:5px;'>Tú: {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#FFE5B4; color:black; padding:10px; border-radius:10px; width:fit-content; margin-bottom:5px;'>Chatbot: {message['content']}</div>", unsafe_allow_html=True)
    st.markdown("-" * 40)