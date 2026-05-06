import cv2
import numpy as np
import json
import platform
import time

# --- 1. CARREGAMENTO DAS CALIBRAÇÕES ---
try:
    with open("cores_cubo.json", "r", encoding="utf-8") as f:
        calibracoes = json.load(f)
    print("Calibracoes carregadas com sucesso.")
except FileNotFoundError:
    print("Erro: o arquivo 'cores_cubo.json' nao foi encontrado.")
    raise SystemExit(1)

cores_visuais = {
    "vermelho": (0, 0, 255),
    "verde": (0, 255, 0),
    "azul": (255, 0, 0),
    "amarelo": (0, 255, 255),
    "laranja": (0, 165, 255),
    "branco": (255, 255, 255),
}

FACE_ORDEM = ["U", "R", "F", "D", "L", "B"]
FACE_NOMES = {
    "U": "Topo",
    "R": "Direita",
    "F": "Frente",
    "D": "Baixo",
    "L": "Esquerda",
    "B": "Tras",
}
JANELA_DETECCAO = "Deteccao Automatica"
JANELA_FACE = "Face Lida (Grade 3x3)"

# Constantes da Grade
TAMANHO_QUADRADO = 100
MARGEM_LEITURA = 30
LIMIAR_LEITURA = 0.15

# Constantes da Região de Interesse (Centro da Tela)
TAMANHO_ROI = 300

def salvar_json(caminho, dados):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4)

# --- FUNÇÕES DE ROTAÇÃO VIRTUAL (Para o Kociemba) ---
def rotacionar_cw(matriz):
    return [list(linha) for linha in zip(*matriz[::-1])]

def rotacionar_ccw(matriz):
    return [list(linha) for linha in list(zip(*matriz))[::-1]]

def rotacionar_180(matriz):
    return [linha[::-1] for linha in matriz[::-1]]

# --- FUNÇÕES AUXILIARES ---
def ordenar_pontos(pts):
    pts = pts.reshape(4, 2)
    ordenados = np.zeros((4, 2), dtype=np.float32)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordenados[0] = pts[np.argmin(soma)]
    ordenados[2] = pts[np.argmax(soma)]
    ordenados[1] = pts[np.argmin(diff)]
    ordenados[3] = pts[np.argmax(diff)]
    return ordenados

def encontrar_camera_c920():
    if platform.system() == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for indice, nome in enumerate(devices):
                if "C920" in nome.upper():
                    return indice
        except ImportError:
            pass
    for i in range(5):
        cap_temp = cv2.VideoCapture(i)
        if cap_temp.isOpened():
            cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            if cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH) == 1920:
                cap_temp.release()
                return i
    return 0

def montar_mascaras(hsv_recortado):
    mascaras = {}
    for nome_cor, limites in calibracoes.items():
        lower = np.array(limites["lower"])
        upper = np.array(limites["upper"])
        mascaras[nome_cor] = cv2.inRange(hsv_recortado, lower, upper)
    return mascaras

def escolher_cor_quadrado(mascaras, x, y, margem=MARGEM_LEITURA, tamanho_quadrado=TAMANHO_QUADRADO):
    melhor_cor = "Nenhum"
    max_pixels = 0
    for nome_cor, mask in mascaras.items():
        roi_mask = mask[y + margem : y + tamanho_quadrado - margem, x + margem : x + tamanho_quadrado - margem]
        pixels = cv2.countNonZero(roi_mask)
        if pixels > max_pixels:
            max_pixels = pixels
            melhor_cor = nome_cor
    area_interna = (tamanho_quadrado - 2 * margem) ** 2
    if max_pixels > (area_interna * LIMIAR_LEITURA) and melhor_cor != "Nenhum":
        return melhor_cor, max_pixels
    return "Nenhum", max_pixels

def ler_grade(face_recortada, margem=30, limiar=0.15):
    hsv_recortado = cv2.cvtColor(face_recortada, cv2.COLOR_BGR2HSV)
    mascaras = montar_mascaras(hsv_recortado)
    grade = []
    face_vis = face_recortada.copy()

    for linha in range(3):
        linha_cores = []
        for coluna in range(3):
            x = coluna * TAMANHO_QUADRADO
            y = linha * TAMANHO_QUADRADO
            melhor_cor, max_pixels = escolher_cor_quadrado(mascaras, x, y, margem, TAMANHO_QUADRADO)
            cor_borda = (255, 255, 255)
            texto = ""
            if max_pixels > ((TAMANHO_QUADRADO - 2 * margem) ** 2 * limiar) and melhor_cor != "Nenhum":
                cor_borda = cores_visuais.get(melhor_cor, (255, 255, 255))
                texto = melhor_cor[:3].upper()
            linha_cores.append(melhor_cor if texto else "Nenhum")
            
            cv2.rectangle(face_vis, (x, y), (x + TAMANHO_QUADRADO, y + TAMANHO_QUADRADO), (100, 100, 100), 1)
            cv2.rectangle(face_vis, (x + margem, y + margem), (x + TAMANHO_QUADRADO - margem, y + TAMANHO_QUADRADO - margem), cor_borda, 2)
            if texto:
                cv2.putText(face_vis, texto, (x + 30, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_borda, 2)
        grade.append(linha_cores)

    centro_cor = grade[1][1]
    cor_centro = cores_visuais.get(centro_cor, (255, 255, 255))
    cv2.drawMarker(face_vis, (150, 150), cor_centro, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    if centro_cor != "Nenhum":
        cv2.putText(face_vis, f"CENTRO: {centro_cor.upper()}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_centro, 2)
    return grade, face_vis, centro_cor

def grade_valida(grade):
    return all(cor != "Nenhum" for linha in grade for cor in linha)

def assinatura_grade(grade):
    return tuple(tuple(linha) for linha in grade)

def mapa_centros(faces):
    cor_para_face = {}
    for face in FACE_ORDEM:
        cor_centro = faces[face][1][1]
        cor_para_face[cor_centro] = face
    return cor_para_face

def converter_faces(faces, cor_para_face):
    convertido = {}
    for face in FACE_ORDEM:
        grade = faces[face]
        nova_grade = []
        for linha in grade:
            nova_linha = []
            for cor in linha:
                nova_linha.append(cor_para_face[cor])
            nova_grade.append(nova_linha)
        convertido[face] = nova_grade
    return convertido

def criar_passos():
    passos = [{"tipo": "captura", "face": "U"}]
    for face in ["F", "R", "D", "L", "B"]:
        passos.append({"tipo": "retorno_u"})
        passos.append({"tipo": "captura", "face": face})
    return passos

# --- 2. CONFIGURACAO DA CAMERA ---
indice_c920 = encontrar_camera_c920()
cap = cv2.VideoCapture(indice_c920, cv2.CAP_DSHOW if platform.system() == "Windows" else 0)

print("Aguardando a camera iniciar...")
time.sleep(1.5)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow(JANELA_DETECCAO, cv2.WINDOW_NORMAL)
cv2.resizeWindow(JANELA_DETECCAO, 1280, 720)

# --- 3. MAQUINA DE ESTADOS ---
passos = criar_passos()
indice_passo = 0
faces_capturadas = {}
cor_centro_u = None

estavel_frames = 10
contagem_estavel = 0
ultima_assinatura = None
pontos_suavizados = None
frames_perdidos = 0
MAX_FRAMES_PERDIDOS = 10

while True:
    success, frame = cap.read()
    if not success:
        break

    # --- DEFINIÇÃO DA REGIÃO DE INTERESSE (ROI) NO CENTRO DA TELA ---
    altura_frame, largura_frame = frame.shape[:2]
    x_roi = (largura_frame - TAMANHO_ROI) // 2
    y_roi = (altura_frame - TAMANHO_ROI) // 2
    
    # Desenha o guia visual para o usuário
    cv2.rectangle(frame, (x_roi, y_roi), (x_roi + TAMANHO_ROI, y_roi + TAMANHO_ROI), (255, 255, 0), 2)
    cv2.putText(frame, "Alinhe o cubo aqui", (x_roi + 10, y_roi + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Recorta apenas o centro da imagem para processamento ultrarrápido
    frame_roi = frame[y_roi:y_roi+TAMANHO_ROI, x_roi:x_roi+TAMANHO_ROI]

    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 30, 80)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contorno_cubo = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 15000 é um bom tamanho dentro da caixa de 600x600
        if area > 15000:
            perimetro = cv2.arcLength(cnt, True)
            aprox = cv2.approxPolyDP(cnt, 0.06 * perimetro, True)
            if len(aprox) == 4:
                x, y, w, h = cv2.boundingRect(aprox)
                aspect_ratio = float(w) / h
                if 0.6 <= aspect_ratio <= 1.4:
                    # Achou o cubo na caixa! Precisamos somar as coordenadas do ROI 
                    # para voltar para as coordenadas do frame original de 1080p
                    contorno_cubo = aprox + [x_roi, y_roi]
                    break

    # --- 4. TRAVAMENTO (LOCK-ON) ---
    if contorno_cubo is not None:
        pts_atuais = ordenar_pontos(contorno_cubo)
        if pontos_suavizados is None:
            pontos_suavizados = pts_atuais
        else:
            pontos_suavizados = (pts_atuais * 0.7) + (pontos_suavizados * 0.3)
        frames_perdidos = 0
    else:
        frames_perdidos += 1
        if frames_perdidos > MAX_FRAMES_PERDIDOS:
            pontos_suavizados = None

    # --- 5. RECORTE PERSPECTIVO E MÁQUINA DE ESTADOS ---
    if pontos_suavizados is not None:
        borda_int = np.int32(pontos_suavizados).reshape((-1, 1, 2))
        cv2.polylines(frame, [borda_int], isClosed=True, color=(0, 255, 0), thickness=3)

        pts_destino = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        matriz_perspectiva = cv2.getPerspectiveTransform(np.float32(pontos_suavizados), pts_destino)
        face_recortada = cv2.warpPerspective(frame, matriz_perspectiva, (300, 300))

        grade, face_vis, centro_cor = ler_grade(face_recortada)

        assinatura_atual = None
        if indice_passo < len(passos):
            passo_atual = passos[indice_passo]
            if passo_atual["tipo"] == "retorno_u":
                if cor_centro_u and centro_cor == cor_centro_u and centro_cor != "Nenhum":
                    assinatura_atual = centro_cor
            else:
                if grade_valida(grade):
                    face_esperada = passo_atual["face"]
                    if face_esperada == "U":
                        assinatura_atual = assinatura_grade(grade)
                    else:
                        if cor_centro_u and grade[1][1] != "Nenhum" and grade[1][1] != cor_centro_u:
                            assinatura_atual = assinatura_grade(grade)

        if assinatura_atual is None:
            contagem_estavel = 0
            ultima_assinatura = None
        else:
            if assinatura_atual == ultima_assinatura:
                contagem_estavel += 1
            else:
                contagem_estavel = 1
                ultima_assinatura = assinatura_atual

        if contagem_estavel >= estavel_frames and indice_passo < len(passos):
            passo_atual = passos[indice_passo]
            if passo_atual["tipo"] == "captura":
                face = passo_atual["face"]
                
                # ROTAÇÃO MATEMÁTICA ANTES DE SALVAR NO JSON
                grade_final = grade
                if face == "R":
                    grade_final = rotacionar_cw(grade)
                elif face == "L":
                    grade_final = rotacionar_ccw(grade)
                elif face == "B":
                    grade_final = rotacionar_180(grade)

                faces_capturadas[face] = grade_final
                
                if face == "U":
                    cor_centro_u = grade_final[1][1]
                print(f"Face capturada e rotacionada: {face}")
            
            indice_passo += 1
            contagem_estavel = 0
            ultima_assinatura = None

        cv2.imshow(JANELA_FACE, face_vis)
    else:
        contagem_estavel = 0
        ultima_assinatura = None
        try:
            cv2.destroyWindow(JANELA_FACE)
        except cv2.error:
            pass

    if indice_passo >= len(passos):
        salvar_json("faces_cubo.json", faces_capturadas)

        cor_para_face = mapa_centros(faces_capturadas)
        faces_convertidas = converter_faces(faces_capturadas, cor_para_face)
        salvar_json("faces_cubo_ref.json", faces_convertidas)

        print("Mapeamento concluido. Arquivos salvos com sucesso!")
        break

    # --- HUD ---
    if indice_passo < len(passos):
        passo_atual = passos[indice_passo]
        if passo_atual["tipo"] == "retorno_u":
            texto_estado = "Volte a face TOPO (U) para a lente"
        else:
            face = passo_atual["face"]
            if face == "U":
                instrucao = "(sua face inicial)"
            elif face == "F":
                instrucao = "-> Tombe o cubo P/ TRAS 1x"
            elif face == "R":
                instrucao = "-> Tombe o cubo P/ ESQUERDA"
            elif face == "D":
                instrucao = "-> Tombe o cubo P/ TRAS 2x" 
            elif face == "L":
                instrucao = "-> Tombe o cubo P/ DIREITA"
            elif face == "B":
                instrucao = "-> Tombe o cubo P/ FRENTE 1x"
            
            texto_estado = f"Mostre {FACE_NOMES[face]} {instrucao}"
    else:
        texto_estado = "Concluido"

    capturadas = " ".join([f for f in FACE_ORDEM if f in faces_capturadas])
    cv2.putText(frame, f"Estado: {texto_estado}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Capturadas: {capturadas}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Estavel: {contagem_estavel}/{estavel_frames}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "[ESC] Sair", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(JANELA_DETECCAO, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()