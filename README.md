# Laboratorio RAG – Gemini + Pinecone + Streamlit

Adaptación del notebook `Practicapdf2026 (1).ipynb`.

## Publicar gratis, sin instalaciones locales

1. En GitHub crea un repositorio nuevo (por ejemplo `laboratorio-rag`).
2. Sube `streamlit_app.py`, `requirements.txt`, `.gitignore` y la carpeta `.streamlit` con su archivo `config.toml`. **Nunca subas tus API keys.**
3. Abre https://share.streamlit.io/ y autentícate con GitHub.
4. Selecciona tu repositorio, rama `main` y archivo principal `streamlit_app.py`.
5. En *Advanced settings → Secrets* ingresa:

   ```toml
   GEMINI_API_KEY = "TU_CLAVE_GEMINI"
   PINECONE_API_KEY = "TU_CLAVE_PINECONE"
   ```
6. Pulsa *Deploy*. Una vez desplegada, comparte la URL `.streamlit.app` con tus estudiantes.

## Flujo didáctico

1. El alumno entra a la app desde el navegador.
2. Carga archivos `.pdf`, `.docx` o `.txt`.
3. Pulsa **Procesar documentos**. La app divide el texto en fragmentos de ~350 caracteres, crea embeddings con `gemini-embedding-001` (3072 dimensiones) y los indexa en Pinecone.
4. Abre **Consultar con IA**, hace preguntas y revisa los fragmentos que se recuperaron.

## Detalles técnicos y límites

- Se conserva el índice `documentos-tae2026-01` del notebook, configurado en 3072 dimensiones y con métrica coseno. Si ya existe, debe ser compatible.
- Se mantiene `gemini-2.5-flash` para la generación de respuestas.
- El nombre de espacio (*namespace*) es único para cada sesión de navegador. Esto evita consultas cruzadas entre sesiones activas, pero **no sustituye la autenticación de usuarios**.
- Las sesiones de Streamlit no son almacenamiento permanente. Los vectores permanecen en Pinecone; nuevas sesiones no tienen forma de recuperar los anteriores desde esta interfaz.
- No se elimina automáticamente el contenido almacenado en Pinecone. El instructor debe gestionar la retención y la eliminación desde Pinecone.
- Una clase completa con claves del instructor puede consumir rápidamente las cuotas gratuitas de Gemini y Pinecone; verifica los límites en los paneles de sus respectivas cuentas.
- Los PDF escaneados necesitan OCR, que esta versión no incluye.
- Recomendación: usar únicamente documentos didácticos sin datos personales ni confidenciales.
- La aplicación no se ha probado contra servicios reales porque las claves no se incluyeron. Valida la primera indexación y consulta después del despliegue.
