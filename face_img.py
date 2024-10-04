import datetime
import math
import cv2
import mediapipe as mp
import json
from pathlib import Path

# Tamanho real da íris (em mm)
TAMANHO_IRIS_REAL = 1.17  # largura em cm
VERSAO = 1.1
PATH_PASTA = "data/"
VAL_ARREDONDAMENTO = 5

# Função para calcular a distância euclidiana entre dois pontos (em pixels)
def calcular_distancia_px(ponto1, ponto2):
    return math.sqrt((ponto2[0] - ponto1[0])**2 + (ponto2[1] - ponto1[1])**2)

def converter_px_to_cm(tam_iris_px, medida_em_px):
    return round((medida_em_px*TAMANHO_IRIS_REAL/tam_iris_px), VAL_ARREDONDAMENTO)

class Ponto:
    def __init__(self, nome, num, x, y, z, exibir):
        self.nome = nome
        self.num = num
        self.x = x
        self.y = y
        self.z = z
        self.exibir = exibir

    # Método para converter o objeto para um dicionário
    def to_dict(self):
        return {"nome": self.nome, "num": self.num, "x": self.x, "y": self.y, "z": self.z, "exibir": self.exibir}

class Analise:
    def __init__(self, versao, data, nomeArquivo, tam_iris_px, tam_iris_cm,
                       altSegundoTerco_px, altSegundoTerco_cm, altTerceiroTerco_px, altTerceiroTerco_cm,
                       alt1TerceiroTerco_px, alt1TerceiroTerco_cm, alt2TerceiroTerco_px, alt2TerceiroTerco_cm,
                       largEntreOlhos_px, largEntreOlhos_cm,
                       largGoniaco_px, largGoniaco_cm, largZigomatico_px, largura_zigomatico_cm,
                       lstPontos):
        self.versaoPrograma = versao
        self.dataAnalise = data
        self.nomeImagemAnalisada = nomeArquivo
        self.iris_px = tam_iris_px
        self.iris_cm = tam_iris_cm
        self.h_2T_px = altSegundoTerco_px
        self.h_2T_cm = altSegundoTerco_cm
        self.h_3T_px = altTerceiroTerco_px
        self.h_3T_cm = altTerceiroTerco_cm
        self.h_3T1_px = alt1TerceiroTerco_px
        self.h_3T1_cm = alt1TerceiroTerco_cm
        self.h_3T2_px = alt2TerceiroTerco_px
        self.h_3T2_cm = alt2TerceiroTerco_cm
        self.l_entreOlhos_px = largEntreOlhos_px
        self.l_entreOlhos_cm = largEntreOlhos_cm
        self.l_goniaco_px = largGoniaco_px
        self.l_goniaco_cm = largGoniaco_cm
        self.l_zigomatico_px = largZigomatico_px
        self.l_zigomatico_cm = largura_zigomatico_cm
        self.lstPontos_dict = lstPontos

    # Método para converter o objeto para um dicionário
    def to_dict(self):
        return {"versaoPrograma": self.versaoPrograma, 
                "dataAnalise": self.dataAnalise, 
                "imagemAnalisada": self.nomeImagemAnalisada, 
                "iris_px": self.iris_px,
                "iris_cm": self.iris_cm,
                "h_2T_px": self.h_2T_px,
                "h_2T_cm": self.h_2T_cm,
                "h_3T_px": self.h_3T_px,
                "h_3T_cm": self.h_3T_cm,
                "h_3T1_px": self.h_3T1_px,
                "h_3T1_cm": self.h_3T1_cm,
                "h_3T2_px": self.h_3T2_px,
                "h_3T2_cm": self.h_3T2_cm,
                "l_entreOlhos_px": self.l_entreOlhos_px,
                "l_entreOlhos_cm": self.l_entreOlhos_cm,
                "l_goniaco_px": self.l_goniaco_px,
                "l_goniaco_cm": self.l_goniaco_cm,
                "l_zigomatico_px": self.l_zigomatico_px,
                "l_zigomatico_cm": self.l_zigomatico_cm,
                "lstPontos": self.lstPontos_dict}

MID_EYES_INDEX = 9
NOSE_BASE_INDEX = 94
CHIN_INDEX = 152
CENTER_MOUNT = 14
LEFT_EYE_INTERNAL = 133
RIGTH_EYE_INTERNAL = 362
# extensao da face
GONIACO_ESQUERDO = 58 
GONIACO_DIREITO = 288
ZIGOMATICO_ESQUERDO = 227
ZIGOMATICO_DIREITO = 447

for arquivo in Path(PATH_PASTA).iterdir():

    lst_pontos_estudo = []
    lst_pontos_estudo.append( Ponto("MID_EYES", MID_EYES_INDEX, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("NOSE_BASE", NOSE_BASE_INDEX, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("CHIN_INDEX", CHIN_INDEX, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("CENTER_MOUNT", CENTER_MOUNT, "", "", "", 0) )
    lst_pontos_estudo.append( Ponto("LEFT_EYE_INTERNAL", LEFT_EYE_INTERNAL, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("RIGTH_EYE_INTERNAL", RIGTH_EYE_INTERNAL, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("GONIACO_ESQUERDO", GONIACO_ESQUERDO, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("GONIACO_DIREITO", GONIACO_DIREITO, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("ZIGOMATICO_ESQUERDO", ZIGOMATICO_ESQUERDO, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("ZIGOMATICO_DIREITO", ZIGOMATICO_DIREITO, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("LEFT_MOUNT", 61, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("RIGTH_MOUNT", 291, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("LEFT_EYE_EXTERNAL", 263, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("RIGTH_EYE_EXTERNAL", 33, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("EYE_LEFT_LOWER_POINT", 145, "", "", "", 0) )
    lst_pontos_estudo.append( Ponto("EYE_RIGTH_LOWER_POINT", 374, "", "", "", 0) )
    lst_pontos_estudo.append( Ponto("SOMBRANCELHA_ESQUERDA", 53, "", "", "", 1) )
    lst_pontos_estudo.append( Ponto("SOMBRANCELHA_DIREITA", 283, "", "", "", 1) )

    #variaveis para lidar com o arquivo
    now_date_time = (datetime.datetime.now()).strftime("%m/%d/%Y, %H:%M:%S")
    stg_date_time = ""
    nome_arquivo = ""
    extensao = ""

    if arquivo.is_file():
        # Obtendo o nome do arquivo e sua extensão
        nome_arquivo = arquivo.stem  # Nome do arquivo sem a extensão
        extensao = arquivo.suffix

        if (extensao == ".jpg"):

                # Inicializando o MediaPipe Face Mesh
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

                # Converter a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo MediaPipe)
                image = cv2.imread(f"{PATH_PASTA}{nome_arquivo}.jpg")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Processando o image com o Face Mesh
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:

                        # Obtendo os pontos chave
                        mid_eyes = face_landmarks.landmark[MID_EYES_INDEX]
                        base_do_nariz = face_landmarks.landmark[NOSE_BASE_INDEX]
                        boca_ao_centro = face_landmarks.landmark[CENTER_MOUNT]
                        queixo = face_landmarks.landmark[CHIN_INDEX]
                        #boca
                        eye_esquerdo_interno = face_landmarks.landmark[LEFT_EYE_INTERNAL]
                        eye_direito_interno = face_landmarks.landmark[RIGTH_EYE_INTERNAL]
                        # extensao da face
                        left_goniaco = face_landmarks.landmark[GONIACO_ESQUERDO]
                        right_goniaco = face_landmarks.landmark[GONIACO_DIREITO]
                        left_zigomatico = face_landmarks.landmark[ZIGOMATICO_ESQUERDO]
                        right_zigomatico = face_landmarks.landmark[ZIGOMATICO_DIREITO]

                        altura, largura, _ = image.shape
                        i = -1
                        for ponto in lst_pontos_estudo:
                            i = i+1
                            p = face_landmarks.landmark[ponto.num]
                            lst_pontos_estudo[i].x = p.x 
                            lst_pontos_estudo[i].y = p.y 
                            lst_pontos_estudo[i].z = p.z 

                            if (ponto.exibir == 1):
                                cv2.circle(image, (int(p.x * largura), int(p.y * altura)), 2, (255, 0, 0), -1)

                        #### IRIS
                        iris1 = face_landmarks.landmark[469]
                        x_iris1, y_iris1 = int(iris1.x * largura), int(iris1.y * altura)

                        iris2 = face_landmarks.landmark[471]
                        x_iris2, y_iris2 = int(iris2.x * largura), int(iris2.y * altura)

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
                        # CALCULOS DISTANCIAS ENTRE OS PONTOS

                        tam_iris_px = calcular_distancia_px((x_iris1, y_iris1), (x_iris2, y_iris2))
                        larg_iris_cm = converter_px_to_cm(tam_iris_px, tam_iris_px)
                        tam_iris_px = round(tam_iris_px, VAL_ARREDONDAMENTO)

                        # Calculando altura 2o terço do rosto - do meio dos olhos a base do nariz
                        altura_2t = calcular_distancia_px((x_mid_eyes, y_mid_eyes), (x_base_do_nariz, y_base_do_nariz))
                        altura_2t_cm = converter_px_to_cm(tam_iris_px, altura_2t)
                        altura_2t = round(altura_2t, VAL_ARREDONDAMENTO)
                            
                        # Calculando altura 3o terço do rosto - da base do nariz ao queixo
                        altura_3t = calcular_distancia_px((x_base_do_nariz, y_base_do_nariz), (x_queixo, y_queixo))
                        altura_3t_cm = converter_px_to_cm(tam_iris_px, altura_3t)
                        altura_3t = round(altura_3t, VAL_ARREDONDAMENTO)

                        # Detalhe do 3o Terco do rosto
                        altura_3t1 = calcular_distancia_px(((x_base_do_nariz, y_base_do_nariz)), (x_boca_ao_centro, y_boca_ao_centro))
                        altura_3t1_cm = converter_px_to_cm(tam_iris_px, altura_3t1)
                        altura_3t1 = round(altura_3t1, VAL_ARREDONDAMENTO)

                        altura_3t2 = calcular_distancia_px((x_boca_ao_centro, y_boca_ao_centro), (x_queixo, y_queixo))
                        altura_3t2_cm = converter_px_to_cm(tam_iris_px, altura_3t2)
                        altura_3t2 = round(altura_3t2, VAL_ARREDONDAMENTO)

                        # Calculando larguras
                        larg_entre_olhos = calcular_distancia_px((x_eye_esquerdo_interno, y_eye_esquerdo_interno), (x_eye_direito_interno, y_eye_direito_interno))
                        larg_entre_olhos_cm = converter_px_to_cm(tam_iris_px, larg_entre_olhos)
                        larg_entre_olhos = round(larg_entre_olhos, VAL_ARREDONDAMENTO)

                        larg_goniaco = calcular_distancia_px((x_left_goniaco, y_left_goniaco), (x_right_goniaco, y_right_goniaco))
                        larg_goniaco_cm = converter_px_to_cm(tam_iris_px, larg_goniaco)
                        larg_goniaco = round(larg_goniaco, VAL_ARREDONDAMENTO)

                        larg_zigomatico = calcular_distancia_px((x_left_zigomatico, y_left_zigomatico), (x_right_zigomatico, y_right_zigomatico))
                        larg_zigomatico_cm = converter_px_to_cm(tam_iris_px, larg_zigomatico)
                        larg_zigomatico = round(larg_zigomatico, VAL_ARREDONDAMENTO)

                        ##########################################
                        # EXIBIR TEXTOS
                        cv2.putText(image, f"{now_date_time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Tam Iris: {larg_iris_cm}cm ({tam_iris_px}px)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Alt Olhos ao Nariz: {altura_2t_cm}cm ({altura_2t}px)", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Alt Nariz ao Queixo: {altura_3t_cm}cm ({altura_3t}px)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Alt Nariz a Boca: {altura_3t1_cm}cm ({altura_3t1}px)", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Alt Boca ao Queixo: {altura_3t2_cm}cm ({altura_3t2}px)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Larg Entre olhos: {larg_entre_olhos_cm}cm ({larg_entre_olhos}px)", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Zigomatico: {larg_zigomatico_cm}cm ({larg_zigomatico}px)", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)
                        cv2.putText(image, f"Goniaco: {larg_goniaco_cm}cm ({larg_goniaco}px)", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 1)

                        ###########################################
                        # ESCREVE LINHAS
                        cv2.line(image, (0, y_mid_eyes), (image.shape[1], y_mid_eyes), (194, 163, 157), 1)
                        cv2.line(image, (0, y_queixo), (image.shape[1], y_queixo), (194, 163, 157), 1)
                        cv2.line(image, (0, y_base_do_nariz), (image.shape[1], y_base_do_nariz), (194, 163, 157), 1)
                        cv2.line(image, (x_base_do_nariz, 0), (x_base_do_nariz, image.shape[1]), (194, 163, 157), 1)

                        #GERA IMAGEM ANALISADA
                        imagem_capturada = image.copy()

                        stg_date_time = now_date_time.replace(" ","")
                        stg_date_time = stg_date_time.replace("/","")
                        stg_date_time = stg_date_time.replace(":","")
                        stg_date_time = stg_date_time.replace(",","")
                        nome_arquivo_analisado = f"{nome_arquivo}_{stg_date_time}.jpg"

                        #JSON
                        lst_pontos_dict = [ponto.to_dict() for ponto in lst_pontos_estudo]
                        analise = Analise(VERSAO, now_date_time, nome_arquivo_analisado, tam_iris_px, TAMANHO_IRIS_REAL, 
                                            altura_2t, altura_2t_cm,  altura_3t, altura_3t_cm, 
                                            altura_3t1, altura_3t1_cm, altura_3t2, altura_3t2_cm, 
                                            larg_entre_olhos, larg_entre_olhos_cm, 
                                            larg_goniaco, larg_goniaco_cm,  larg_zigomatico, larg_zigomatico_cm, 
                                            lst_pontos_dict)

                        with open(f"{PATH_PASTA}{nome_arquivo}_{stg_date_time}.json", "w") as arquivo_json:
                            json.dump(analise.to_dict(), arquivo_json, indent=4)

                        cv2.imwrite(f"{PATH_PASTA}{nome_arquivo_analisado}", imagem_capturada)
                        cv2.destroyAllWindows()

                else:
                    print("Nenhum rosto detectado.")


