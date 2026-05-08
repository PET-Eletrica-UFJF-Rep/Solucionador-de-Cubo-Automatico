import cv2
import numpy as np
import json
import platform
import time

# --- 1. CARREGAMENTO DAS CALIBRAÇÕES ---
try:
    with open('cores_cubo.json', 'r', encoding='utf-8') as f:
        calibracoes = json.load(f)
    print("✅ Calibrações carregadas com sucesso!")
except FileNotFoundError:
    print("❌ Erro: O arquivo 'cores_cubo.json' não foi encontrado.")
    exit()

cores_visuais = {
    'vermelho': (0, 0, 255),
    'verde': (0, 255, 0),
    'azul': (255, 0, 0),
    'amarelo': (0, 255, 255),
    'laranja': (0, 165, 255),
    'branco': (255, 255, 255)
}

# --- FUNÇÃO AUXILIAR: ORDENAR PONTOS (Resolve o problema do ângulo) ---
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

# --- 2. CONFIGURAÇÃO DA CÂMERA ---
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

indice_c920 = encontrar_camera_c920()
cap = cv2.VideoCapture(indice_c920, cv2.CAP_DSHOW if platform.system() == "Windows" else 0)

print("Aguardando a câmera iniciar...")
time.sleep(1.5)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Deteccao Automatica", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteccao Automatica", 1280, 720)

# Variáveis de Memória e Otimização
pontos_suavizados = None
frames_perdidos = 0
MAX_FRAMES_PERDIDOS = 10
escala = 0.5  # Processa o contorno na metade da resolução para ser muito rápido

while True:
    success, frame = cap.read()
    if not success:
        break

    # --- 3. DETECÇÃO DA SILHUETA EXTERNA DO CUBO ---
    frame_reduzido = cv2.resize(frame, (0, 0), fx=escala, fy=escala)
    gray = cv2.cvtColor(frame_reduzido, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Canny acha as bordas. Em cubos stickerless, a borda externa contra a mesa é a mais forte.
    edges = cv2.Canny(blurred, 30, 80)
    
    kernel = np.ones((7, 7), np.uint8)
    # Fechamento ajuda a ignorar as pequenas linhas de divisão entre as peças plásticas
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contorno_cubo = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (8000 * (escala ** 2)):  # Cubo precisa ter um tamanho razoável
            perimetro = cv2.arcLength(cnt, True)
            # 0.06 de tolerância ajuda a pegar as quinas arredondadas dos speedcubes
            aprox = cv2.approxPolyDP(cnt, 0.06 * perimetro, True)
            
            if len(aprox) == 4:
                x, y, w, h = cv2.boundingRect(aprox)
                aspect_ratio = float(w) / h
                if 0.6 <= aspect_ratio <= 1.4: # Garante que não é um retângulo esticado
                    contorno_cubo = aprox / escala # Reverte a escala para bater com o frame 1080p
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

    # --- 5. RECORTE PERSPECTIVO E LEITURA DA GRADE ---
    if pontos_suavizados is not None:
        # Desenha a borda da detecção na tela original
        borda_int = np.int32(pontos_suavizados).reshape((-1, 1, 2))
        cv2.polylines(frame, [borda_int], isClosed=True, color=(0, 255, 0), thickness=3)

        # Achata a imagem do cubo para um quadrado perfeito 300x300
        pts_destino = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        matriz_perspectiva = cv2.getPerspectiveTransform(np.float32(pontos_suavizados), pts_destino)
        face_recortada = cv2.warpPerspective(frame, matriz_perspectiva, (300, 300))

        hsv_recortado = cv2.cvtColor(face_recortada, cv2.COLOR_BGR2HSV)
        
        mascaras = {}
        for nome_cor, limites in calibracoes.items():
            lower = np.array(limites['lower'])
            upper = np.array(limites['upper'])
            mascaras[nome_cor] = cv2.inRange(hsv_recortado, lower, upper)

        # Como a imagem tem 300x300, sabemos que cada peça tem 100x100.
        tamanho_quadrado = 100
       
        # Vamos olhar APENAS para os 40x40 pixels bem no centro da peça, ignorando as sombras da divisão física!
        margem = 30 
        area_interna = (tamanho_quadrado - 2*margem) ** 2 

        centro_cor = "Nenhum"
        
        for linha in range(3):
            for coluna in range(3):
                x = coluna * tamanho_quadrado
                y = linha * tamanho_quadrado
                
                melhor_cor = "Nenhum"
                max_pixels = 0
                
                for nome_cor, mask in mascaras.items():
                    # Recorta apenas a área de segurança (miolinho)
                    roi_mask = mask[y+margem:y+tamanho_quadrado-margem, x+margem:x+tamanho_quadrado-margem]
                    pixels = cv2.countNonZero(roi_mask)
                    
                    if pixels > max_pixels:
                        max_pixels = pixels
                        melhor_cor = nome_cor

                cor_borda = (255, 255, 255)
                texto = ""
                
                # Diminui a exigência: se 15% do miolinho for da cor, já aceitamos
                if max_pixels > (area_interna * 0.15) and melhor_cor != "Nenhum":
                    cor_borda = cores_visuais.get(melhor_cor, (255, 255, 255))
                    texto = melhor_cor[:3].upper()

                if linha == 1 and coluna == 1:
                    centro_cor = melhor_cor if texto else "Nenhum"

                # Desenha o quadrado da peça toda
                cv2.rectangle(face_recortada, (x, y), (x + tamanho_quadrado, y + tamanho_quadrado), (100, 100, 100), 1)
                
                # Desenha o quadradinho de leitura (miolo) com a cor detectada
                cv2.rectangle(face_recortada, (x+margem, y+margem), (x+tamanho_quadrado-margem, y+tamanho_quadrado-margem), cor_borda, 2)

                if linha == 1 and coluna == 1:
                    # Destaca o centro da face para facilitar a leitura
                    cv2.drawMarker(face_recortada, (x + 50, y + 50), cor_borda, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

                if texto:
                    cv2.putText(face_recortada, texto, (x + 30, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_borda, 2)

        if centro_cor != "Nenhum":
            cor_texto_centro = cores_visuais.get(centro_cor, (255, 255, 255))
            cv2.putText(face_recortada, f"CENTRO: {centro_cor.upper()}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_texto_centro, 2)

        cv2.imshow("Face Lida (Grade 3x3)", face_recortada)
    else:
        try:
            cv2.destroyWindow("Face Lida (Grade 3x3)")
        except:
            pass

    cv2.putText(frame, "Caca-Pecas Automatico", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, "Caca-Pecas Automatico", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "[ESC] Sair", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Deteccao Automatica", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()