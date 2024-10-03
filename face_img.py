import datetime
import math
import cv2
import mediapipe as mp

# Tamanho real da íris (em mm)
TAMANHO_IRIS_REAL = 1.17  # largura em cm

# Pontos de interesse:
MID_EYES_INDEX = 9 #168
NOSE_BASE_INDEX = 94
CHIN_INDEX = 152
CENTER_MOUNT = 14
# boca
LEFT_MOUNT = 61
RIGTH_MOUNT = 291
# olhos
LEFT_EYE_INTERNAL = 133
LEFT_EYE_EXTERNAL = 263
RIGTH_EYE_INTERNAL = 362
RIGTH_EYE_EXTERNAL = 33
EYE_LEFT_LOWER_POINT = 145
EYE_RIGTH_LOWER_POINT = 374
SOMBRANCELHA_ESQUERDA = 53
SOMBRANCELHA_DIREITA = 283
# extensao da face
GONIACO_ESQUERDO = 58 
GONIACO_DIREITO = 288
ZIGOMATICO_ESQUERDO = 227
ZIGOMATICO_DIREITO = 447

# Função para calcular a distância euclidiana entre dois pontos (em pixels)
def calcular_distancia_px(ponto1, ponto2):
    return math.sqrt((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2)

def converter_px_to_cm(tam_iris_px, medida_em_px):
    return round((medida_em_px*TAMANHO_IRIS_REAL/tam_iris_px),2)

# Inicializando o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    refine_landmarks=True)

# Carregar a imagem
IMG_INFO = "data/teste"
image = cv2.imread(f"{IMG_INFO}.jpg")

# Converter a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo MediaPipe)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Processando o image com o Face Mesh
results = face_mesh.process(image_rgb)

now_date_time = (datetime.datetime.now()).strftime("%m/%d/%Y, %H:%M:%S")
stg_date_time = ""

if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Obtendo os pontos chave
            mid_eyes = face_landmarks.landmark[MID_EYES_INDEX]
            base_do_nariz = face_landmarks.landmark[NOSE_BASE_INDEX]
            boca_ao_centro = face_landmarks.landmark[CENTER_MOUNT]
            queixo = face_landmarks.landmark[CHIN_INDEX]
            #boca
            boca_a_esquerda = face_landmarks.landmark[LEFT_MOUNT]
            boca_a_direita = face_landmarks.landmark[RIGTH_MOUNT]
            # olhos
            eye_esquerdo_interno = face_landmarks.landmark[LEFT_EYE_INTERNAL]
            eye_esquerdo_externo = face_landmarks.landmark[LEFT_EYE_EXTERNAL]
            eye_direito_interno = face_landmarks.landmark[RIGTH_EYE_INTERNAL]
            eye_direito_externo = face_landmarks.landmark[RIGTH_EYE_EXTERNAL]
            eye_esquerdo_embaixo = face_landmarks.landmark[EYE_LEFT_LOWER_POINT]
            eye_direito_embaixo = face_landmarks.landmark[EYE_RIGTH_LOWER_POINT]
            somb_esquerda = face_landmarks.landmark[SOMBRANCELHA_ESQUERDA]
            somb_direita = face_landmarks.landmark[SOMBRANCELHA_DIREITA]
            # extensao da face
            left_goniaco = face_landmarks.landmark[GONIACO_ESQUERDO]
            right_goniaco = face_landmarks.landmark[GONIACO_DIREITO]
            left_zigomatico = face_landmarks.landmark[ZIGOMATICO_ESQUERDO]
            right_zigomatico = face_landmarks.landmark[ZIGOMATICO_DIREITO]

            altura, largura, _ = image.shape

            pontos = [mid_eyes, base_do_nariz, queixo, boca_a_esquerda, boca_a_direita, boca_ao_centro, 
                      eye_esquerdo_embaixo, eye_direito_embaixo, somb_esquerda, somb_direita,
                      eye_esquerdo_interno, eye_direito_interno, eye_esquerdo_externo, eye_direito_externo,
                      left_goniaco, right_goniaco, left_zigomatico, right_zigomatico]

            for p in pontos:
                cv2.circle(image, (int(p.x * largura), int(p.y * altura)), 2, (255, 0, 0), -1)

            #### IRIS
            iris1 = face_landmarks.landmark[469]
            x_iris1, y_iris1 = int(iris1.x * largura), int(iris1.y * altura)

            iris2 = face_landmarks.landmark[471]
            x_iris2, y_iris2 = int(iris2.x * largura), int(iris2.y * altura)

            tam_iris_px = x_iris1 - x_iris2

            ####
            
            ###########################################
            # MAPEANDO OS PRINCIPAIS PONTOS
            # Convertendo coordenadas normalizadas (entre 0 e 1) para pixels
            x_mid_eyes, y_mid_eyes = int(mid_eyes.x * largura), int(mid_eyes.y * altura)
            x_base_do_nariz, y_base_do_nariz = int(base_do_nariz.x * largura), int(base_do_nariz.y * altura)
            x_boca_ao_centro, y_boca_ao_centro = int(boca_ao_centro.x * largura), int(boca_ao_centro.y * altura)
            x_queixo, y_queixo = int(queixo.x * largura), int(queixo.y * altura)
            x_eye_esquerdo_interno, y_eye_esquerdo_interno = int(eye_esquerdo_interno.x * largura), int(eye_esquerdo_interno.y * altura)
            x_eye_direito_interno, y_eye_direito_interno = int(eye_direito_interno.x * largura), int(eye_direito_interno.y * altura)
            x_left_goniaco, y_left_goniaco = int(left_goniaco.x * largura), int(left_goniaco.y * altura)
            x_right_goniaco, y_right_goniaco = int(right_goniaco.x * largura), int(right_goniaco.y * altura)
            x_left_zigomatico, y_left_zigomatico = int(left_zigomatico.x * largura), int(left_zigomatico.y * altura)
            x_right_zigomatico, y_right_zigomatico = int(right_zigomatico.x * largura), int(right_zigomatico.y * altura)

            ###########################################
            # CALCULOS

            # Calculando altura 2o terço do rosto - do meio dos olhos a base do nariz
            altura_2t_rosto = y_base_do_nariz - y_mid_eyes
            
            # Calculando altura 3o terço do rosto - da base do nariz ao queixo
            altura_3t_rosto = y_queixo - y_base_do_nariz
            
            # Calculando altura 3o terço do rosto - da base do nariz ao queixo
            altura_3t1_rosto = y_boca_ao_centro - y_base_do_nariz
            altura_3t2_rosto = y_queixo - y_boca_ao_centro

            # Calculando larguras
            largura_entre_olhos = x_eye_direito_interno - x_eye_esquerdo_interno
            largura_goniaco_rosto = x_right_goniaco - x_left_goniaco
            largura_zigomatico_rosto = x_right_zigomatico - x_left_zigomatico

            ##########################################
            # EXIBIR TEXTOS
            # Exibir texto no image
            cv2.putText(image, f"{now_date_time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Tam Iris: {converter_px_to_cm(tam_iris_px, tam_iris_px)}cm ({tam_iris_px}px)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Alt Olhos ao Nariz: {converter_px_to_cm(tam_iris_px, altura_2t_rosto)}cm ({altura_2t_rosto}px)", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Alt Nariz ao Queixo: {converter_px_to_cm(tam_iris_px, altura_3t_rosto)}cm ({altura_3t_rosto}px)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Alt Nariz a Boca: {converter_px_to_cm(tam_iris_px, altura_3t1_rosto)}cm ({altura_3t1_rosto}px)", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Alt Boca ao Queixo: {converter_px_to_cm(tam_iris_px, altura_3t2_rosto)}cm ({altura_3t2_rosto}px)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Larg Entre olhos: {converter_px_to_cm(tam_iris_px, largura_entre_olhos)}cm ({largura_entre_olhos}px)", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Zigomatico: {converter_px_to_cm(tam_iris_px, largura_zigomatico_rosto)}cm ({largura_zigomatico_rosto}px)", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
            cv2.putText(image, f"Goniaco: {converter_px_to_cm(tam_iris_px, largura_goniaco_rosto)}cm ({largura_goniaco_rosto}px)", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)

            ###########################################
            # ESCREVE LINHAS
            cv2.line(image, (0, y_mid_eyes), (image.shape[1], y_mid_eyes), (194, 163, 157), 1)
            cv2.line(image, (0, y_queixo), (image.shape[1], y_queixo), (194, 163, 157), 1)
            cv2.line(image, (0, y_base_do_nariz), (image.shape[1], y_base_do_nariz), (194, 163, 157), 1)
            cv2.line(image, (x_base_do_nariz, 0), (x_base_do_nariz, image.shape[1]), (194, 163, 157), 1)

            cv2.imshow('Imagem analisada', image)
            imagem_capturada = image.copy()

            stg_date_time = now_date_time.replace(" ","")
            stg_date_time = stg_date_time.replace("/","")
            stg_date_time = stg_date_time.replace(":","")
            stg_date_time = stg_date_time.replace(",","")
            nome_arquivo = f"{IMG_INFO}_analisada_{stg_date_time}.jpg"
            cv2.imwrite(nome_arquivo, imagem_capturada)
            print(f"Imagem salva como {nome_arquivo}")

            cv2.waitKey(0)
            cv2.destroyAllWindows()

else:
    print("Nenhum rosto detectado.")

