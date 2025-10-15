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
import csv 
import os

# Inicialización de la app
st.set_page_config(page_title="Chatbot de Reservas - Mejorado", page_icon="🤖", layout="centered")
st.title("Chatbot de Reservas - Mejorado")

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
    "saludo": [
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "¿qué tal?"
    ],
    "despedida": [
        "adiós", "hasta luego", "nos vemos", "chao", "bye"
    ],
    "reservar_mesa": [
        "Quiero reservar una mesa",
        "Reservar para 2 personas mañana por la noche",
        "Necesito una mesa para 4 personas a las 20:00",
        "Me gustaría reservar una mesa el sábado a las 21",
        "Reserva para 3 el 10/10 a las 19:30"
    ],
    "cancelar_reserva": [
        "Quiero cancelar mi reserva",
        "Cancelar mesa",
        "Anular reserva",
        "No podré ir a la reserva"
    ],
    "pregunta_menu": [
        "¿Qué menú tienen?",
        "Mostrar menú",
        "¿Cuál es el menú del día?",
        "¿Tienen opciones vegetarianas?"
    ],
    "pregunta_horario": [
        "¿Cuál es el horario?", "¿A qué hora abren?", "Horario de atención"
    ],
    "confirmacion": [
        "sí", "si", "claro", "perfecto", "confirmar"
    ],
    "negacion": [
        "no", "nop", "no gracias", "ahora no"
    ]
}

# ----- Guardar las reservas en CSV -----

# Archivo CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "reservas.csv")

# Función para guardar reservas en un archivo CSV
def save_reservations_to_csv(reservation):
    # Comprobar si el archivo ya existe
    file_exists = os.path.isfile(CSV_FILE)
    
    # Abrir el archivo en modo append
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Si el archivo no existe, escribir la cabecera
        if not file_exists:
            writer.writerow(["num_personas", "date", "time", "created_at"])
        # Escribir la reserva
        writer.writerow([reservation['num_personas'], reservation['date'], reservation['time'], reservation['created_at']])

# ----- Leer reservas desde el archivo CSV -----

# Función para cargar reservas desde un archivo CSV
def load_reservations_from_csv():
    reservations = []
    # Comprobar si el archivo existe
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convertir tipos de datos
                row['num_personas'] = int(row['num_personas'])
                row['date'] = row['date']
                row['time'] = row['time']
                row['created_at'] = row['created_at']
                reservations.append(row)
    return reservations

# Precalcular embeddings de ejemplos
examples_embeddings = {}
# Calcular embeddings para cada intent
for intent, examples in intents.items():
    # Verificar si hay ejemplos disponibles
    if len(examples) > 0:
        # Calcular embeddings de los ejemplos
        embeddings = embed_model.encode(examples, convert_to_tensor=True)
        # Guardar en un diccionario
        examples_embeddings[intent] = embeddings
    else:
        # Si no hay ejemplos, asignar None
        examples_embeddings[intent] = None

# ----- Estado de la sesión -----
# Historial de la conversación
if 'history' not in st.session_state:
    st.session_state.history = []
# Contexto actual
if 'pending_action' not in st.session_state:
    # pending_action ejemplo: {"action":"reservar_mesa", "slots": {"num_personas": None, "date": None, "time": None}}
    st.session_state.pending_action = None
# Lista de reservas realizadas
if 'reservations' not in st.session_state:
    st.session_state.reservations = load_reservations_from_csv()

# ----- Funciones de NLP -----
SIMILARITY_THRESHOLD = 0.55  # Umbral de similitud para aceptar un intent (ajustable)


# Función de clasificación de intents
def predict_intent(user_input):
    # Calcular embedding del input del usuario
    input_embedding = embed_model.encode(user_input, convert_to_tensor=True)
    max_sim = -1
    best_intent = None
    
    # Comparar con cada intent
    for intent, embeddings in examples_embeddings.items():
        if embeddings is None:
            continue
        sim_scores = util.cos_sim(input_embedding, embeddings)
        sim_score = sim_scores.max().item()
        if sim_score > max_sim:
            max_sim = sim_score
            best_intent = intent
    # Fallback si no supera el umbral
    if max_sim < SIMILARITY_THRESHOLD:
        return "fallback", max_sim
    return best_intent, max_sim

# Función de extracción de entidades
def extract_entities(user_input):
    doc = ner_model(user_input)
    entities = {}
    # SpaCy NER (LABEL -> text)
    for ent in doc.ents:
        # Guardar varios valores posibles (si hay varios de la misma etiqueta)
        if ent.label_ in entities:
            if isinstance(entities[ent.label_], list):
                entities[ent.label_] += f" | {ent.text}"
            else:
                entities[ent.label_] = ent.text
    
    # Regex para extraer el número de personas (ej: "para 2 personas", "mesa para 4", etc.)
    # Primero números escritos en dígitos
    match = re.search(r'\b(?:para\s+)?(\d{1,2})\s*(?:personas|pers|pax)?\b', user_input, flags=re.IGNORECASE)
    if match:
        entities['NUM_PERSONAS'] = match.group(1)
        
    # Regex para horas HH:MM o H:MM o H (ej: 20:00, 9:30, 21)
    match_time = re.search(r'\b([01]?\d|2[0-3])[:hH]?([0-5]\d)?\b', user_input)
    if match_time:
        # Construir hora legible
        h = match_time.group(1)
        mm = match_time.group(2) if match_time.group(2) else "00"
        entities['TIME'] = f"{h}:{mm}"
    
    # Regex para fechas (formato dd/mm, dd-mm, nombres de días)
    match_date = re.search(r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b', user_input)
    if match_date:
        entities['DATE'] = match_date.group(1)
        
    # También date o time pueden ser reconocidos por spaCy, así que no los sobreescribimos si ya existen
    if 'DATE' in entities and 'DATE' in entities:
        pass
    
    return entities


# ----- Lógica de negocio: slot filling y manejo de reservas -----
def start_reservation_flow(entities):
    # Inicializar pending_action con slots vacíos o rellenados si ya se tienen
    slots = {"num_personas": None, "date": None, "time": None}
    if entities.get("NUM_PERSONAS"):
        slots['num_personas'] = entities['NUM_PERSONAS']
    elif entities.get("CARDINAL"):
        slots['num_personas'] = entities['CARDINAL']
    if entities.get("DATE"):
        slots['date'] = entities['DATE']
    if entities.get("TIME"):
        slots['time'] = entities['TIME']
    st.session_state.pending_action = {"action": "reservar_mesa", "slots": slots}
    
def fill_slot_from_answer(answer):
    # Intent: intentar llenar el slot vacío con el último texto del usuario
    # Buscamos patrones de número/hora/fecha
    ent = extract_entities(answer)
    slots = st.session_state.pending_action['slots']
    changed = False
    if not slots['num_personas']:
        if ent.get('NUM_PERSONAS'):
            slots['num_personas'] = ent['NUM_PERSONAS']
            changed = True
        elif ent.get('CARDINAL'):
            slots['num_personas'] = ent['CARDINAL']
            changed = True
    if not slots['time'] and ent.get('TIME'):
        slots['time'] = ent['TIME']; changed = True
    if not slots['date'] and ent.get('DATE'):
        slots['date'] = ent['DATE']; changed = True
    st.session_state.pending_action['slots'] = slots
    return changed

def finalize_reservation():
    slots = st.session_state.pending_action['slots']
    # Validar mínimos
    num = slots.get('num_personas') or "1"
    date = slots.get('date') or "fecha no especificada"
    time = slots.get('time') or "hora no especificada"
    
    # Guardar la reserva en la memoria 
    reservation = {
        "num_personas": num, 
        "date": date, 
        "time": time,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Guardar en CSV
    save_reservations_to_csv(reservation)
    
    st.session_state.reservations.append(reservation)
    st.session_state.pending_action = None
    return reservation


# ----- Generación de respuestas -----
def generate_response(intent, entities, user_input):
    # Si hay flujo pendiente (slot filling), priorizarlo
    if st.session_state.pending_action:
        # Intent: intentar rellenar slots con la respuesta del usuario
        filled = fill_slot_from_answer(user_input)
        slots = st.session_state.pending_action['slots']
        # Preguntar por los slots que faltan
        if not slots['num_personas']:
            return "¿Para cuántas personas es la reserva?"
        if not slots['date']:
            return "¿Para qué fecha te gustaría hacer la reserva?"
        if not slots['time']:
            return "¿A qué hora te gustaría reservar la mesa?"
        # Si ya están todos los slots, finalizar la reserva
        reservation = finalize_reservation()
        return f"¡Reserva confirmada para {reservation['num_personas']} personas el {reservation['date']} a las {reservation['time']}! ¿Necesitas algo más?"
    
    # Flujo normal cuando no hay pending_action
    if intent == "saludo":
        return random.choice(["¡Hola! ¿Deseas reservar una mesa?", "¡Buenas! ¿En qué puedo ayudarte hoy?"])
    elif intent == "despedida":
        return random.choice(["¡Hasta luego!", "Que tengas un buen día."])
    elif intent == "reservar_mesa":
        # Iniciar flujo de reserva
        start_reservation_flow(entities)
        # Si ya hay slots completos, finalizar directamente
        slots = st.session_state.pending_action['slots']
        if slots['num_personas'] and slots['date'] and slots['time']:
            reservation = finalize_reservation()
            return f"¡Reserva confirmada para {reservation['num_personas']} personas el {reservation['date']} a las {reservation['time']}! ¿Necesitas algo más?"
        # Si faltan slots, preguntar por el primero que falte (stop filling)
        if not slots['num_personas']:
            return "¿Para cuántas personas es la reserva?"
        if not slots['date']:
            return "¿Para qué fecha te gustaría hacer la reserva?"
        if not slots['time']:
            return "¿A qué hora te gustaría reservar la mesa?"
    elif intent == "cancelar_reserva":
        # Logica simple: cancelar la última reserva
        if st.session_state.reservations:
            removed = st.session_state.reservations.pop()
            return f"Tu reserva para {removed['num_personas']} personas el {removed['date']} a las {removed['time']} ha sido cancelada."
        else:
            return "No tienes reservas para cancelar."
    elif intent == "pregunta_menu":
        return "Nuestro menú incluye opciones vegetarianas y sin gluten. ¿Quieres que te envíe el menú completo por email?"
    elif intent == "pregunta_horario":
        return "Nuestro horario de atención es de lunes a domingo de 12:00 a 23:00."
    elif intent == "fallback":
        return "Lo siento, no he entendido tu mensaje. ¿Podrías reformularlo?"
    else:
        return "Lo siento, no he entendido tu solicitud. ¿Podrías aclararlo?"
    

# ----- Interfaz con Streamlit -----
# Interfaz de usuario
chat_placeholder = st.container()
user_input = st.chat_input("Escribe tu mensaje aquí:")

if user_input:
    # Guardar mensaje del usuario
    st.session_state.history.append({"role": "user", "content": user_input, "timestamp": datetime.utcnow().isoformat()})
    
    # Si hay un flujo pendiente, llamamos a generate_response directamente
    if st.session_state.pending_action:
        bot_response = generate_response(None, {}, user_input)
        st.session_state.history.append({"role": "bot", "content": bot_response})
    else:
        # Predecir intent y extraer entidades
        intent, sim = predict_intent(user_input)
        entities = extract_entities(user_input)
        bot_response = generate_response(intent, entities, user_input)
        # Añadir info de depuración al historial (opcional)
        st.session_state.history.append({"role": "bot", "content": bot_response, "meta": {"intent": intent, "sim": round(sim, 3), "entities": entities}})
    
# Mostrar el historial de la conversación
with chat_placeholder :
    for message in st.session_state.history[-16:]: # Mostrar solo los últimos 16 mensajes
        if message['role'] == 'user':   
            st.markdown(f"Tú: {message['content']}")
        else:
            st.markdown(f"Chatbot: {message['content']}")
            # Mostrar metadatos de depuración si existen modo DEBUG
            DEBUG = False # Cambiar a True para ver detalles
            if 'meta' in message and DEBUG:
                meta = message['meta']
                st.caption(f"Intent: {meta.get('intent')} · sim: {meta.get('sim')} · entidades: {meta.get('entities')}")
        st.markdown("-" * 40)

# Mostrar reservas actuales (para verificación)
st.markdown("Reservas actuales:")
if st.session_state.reservations:
    for r in st.session_state.reservations:
        st.markdown(f"- {r['num_personas']} personas el {r['date']} a las {r['time']}")
    
else:
    st.info("No hay reservas realizadas.")


