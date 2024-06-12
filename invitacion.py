import cohere
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from prodiapy import Prodia
from io import BytesIO

# Configuración de la API de Cohere
co = cohere.Client(api_key="lb6MoszC0F7SBzh5rWUdNF5y4P5CEy2iAvhPDNyJ")

response = co.generate(
    prompt=f"""te doy un ejemplo con las carateristicas que quiero que generes:
    Queridos amigos, los invitamos a celebrar la Navidad con nosotros. Cena, risas y regalos el 24/12. Confirmar asistencia antes del 15/12.
    devuelve SOLO la invitacion. Cada 4 palabras salto de linea.
    """,
    max_tokens=30,
    temperature=0.1,
)

texto = response.generations[0].text.strip()

# Configuración de la API de Prodia
prodia = Prodia(
    api_key="77af6ba4-3d1a-4d5f-80bc-264e0083b674"
)

job = prodia.sd.generate(prompt="christmas landscape")
result = prodia.wait(job)

img_url = result.image_url

# Realizar la solicitud GET para obtener la imagen
response = requests.get(img_url)
response.raise_for_status()  # Asegurarse de que la solicitud fue exitosa

# Convertir el contenido de la respuesta a un array de NumPy
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

# Leer la imagen con OpenCV
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Verificar si la imagen se ha leído correctamente
if image is not None:
    # Escribir el texto en la imagen
    texted_image = cv2.putText(
        img=np.copy(image), 
        text=texto, 
        org=(10, 450), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1, 
        color=(0, 0, 0), 
        thickness=3
    )

    # Mostrar la imagen con el texto utilizando Matplotlib
    plt.imshow(cv2.cvtColor(texted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Ocultar los ejes
    plt.show()
else:
    print("Error al leer la imagen con OpenCV")
