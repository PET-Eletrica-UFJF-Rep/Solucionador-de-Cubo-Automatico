import cv2
import numpy as np
import json
import platform
import time
import os # <--- Adicionado para checar se o arquivo existe

# Função vazia necessária para as trackbars do OpenCV
def empty(a):
    pass

# Função para encontrar automaticamente a Logitech C920
def encontrar_camera_c920():
    if platform.system() == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
            graph = FilterGraph()
            devices = graph.get_input_devices()
            
            for indice, nome in enumerate(devices):
                if "C920" in nome.upper():
                    print(f"✅ Logitech C920 encontrada no índice {indice} ({nome})")
                    return indice
        except ImportError:
            print("⚠️ Biblioteca 'pygrabber' não encontrada. Tentando método alternativo...")
    
    print("Buscando câmera por tentativa e erro (buscando suporte a 1080p)...")
    for i in range(5):
        cap_temp = cv2.VideoCapture(i)
        if cap_temp.isOpened():
            cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
            cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
            largura_real = cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)
            cap_temp.release() 
            
            if largura_real == 1920:
                print(f"✅ Câmera Full HD (possível C920) encontrada no índice {i}")
                return i
                
    print("❌ C920 não encontrada automaticamente. Usando câmera padrão (0).")
    return 0

# Criação da janela e das Trackbars de calibração
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 250)
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

# --- MODIFICAÇÃO PRINCIPAL: Carregar JSON existente ---
arquivo_json = 'cores_cubo.json'
calibracoes = {}

if os.path.exists(arquivo_json):
    with open(arquivo_json, 'r', encoding='utf-8') as f:
        calibracoes = json.load(f)
    mensagem_status = "JSON carregado! Ajuste e regrave a cor desejada."
else:
    mensagem_status = "Novo JSON. Ajuste as barras e grave a cor."
# --------------------------------------------------------

# Conexão com a câmera
indice_c920 = encontrar_camera_c920()

if platform.system() == "Windows":
    # cv2.CAP_DSHOW melhora a compatibilidade com a C920 no Windows
    cap = cv2.VideoCapture(indice_c920, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(indice_c920)

print("Aguardando a câmera iniciar...")
time.sleep(1.5) # Dá tempo para a câmera "acordar"

# Força resolução 1080p para maior precisão
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Erro crítico: O OpenCV conectou na câmera, mas não conseguiu puxar a imagem.")
        print("Verifique se outro programa (Discord, OBS, Zoom, navegador) não está com ela aberta!")
        break

    # Converte de BGR para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lê as posições atuais das trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    lower_color = np.array([h_min, s_min, v_min])
    upper_color = np.array([h_max, s_max, v_max])

    # Cria a máscara para limiarização
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Desenha os contornos da cor sendo testada no momento
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Filtra os ruídos
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Interface de Texto na Tela (HUD)
    cv2.putText(frame, "Gravar Cor: [V]ermelho [G]Verde [A]zul [M]Amarelo [L]Laranja [B]Branco", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Salvar JSON: Aperte [S] | Sair: Aperte [ESC]", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {mensagem_status}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Webcam Original", frame)
    cv2.imshow("Mascara (Limiarizacao)", mask)

    # Lógica do teclado para salvar as cores no dicionário
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # 27 = Tecla ESC
        break
        
    valores_atuais = {
        "lower": [int(h_min), int(s_min), int(v_min)],
        "upper": [int(h_max), int(s_max), int(v_max)]
    }

    if key == ord('v'):
        calibracoes['vermelho'] = valores_atuais
        mensagem_status = "Vermelho atualizado!"
    elif key == ord('g'):
        calibracoes['verde'] = valores_atuais
        mensagem_status = "Verde atualizado!"
    elif key == ord('a'):
        calibracoes['azul'] = valores_atuais
        mensagem_status = "Azul atualizado!"
    elif key == ord('m'):
        calibracoes['amarelo'] = valores_atuais
        mensagem_status = "Amarelo atualizado!"
    elif key == ord('l'):
        calibracoes['laranja'] = valores_atuais
        mensagem_status = "Laranja atualizado!"
    elif key == ord('b'):
        calibracoes['branco'] = valores_atuais
        mensagem_status = "Branco atualizado!"
        
    elif key == ord('s'):
        with open(arquivo_json, 'w', encoding='utf-8') as f:
            json.dump(calibracoes, f, indent=4)
        mensagem_status = "Arquivo 'cores_cubo.json' SALVO!"

cap.release()
cv2.destroyAllWindows()