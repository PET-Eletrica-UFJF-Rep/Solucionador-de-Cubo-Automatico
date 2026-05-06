import json
import sys
import kociemba

FACE_ORDEM = ["U", "R", "F", "D", "L", "B"]
FACE_NOMES = {
    "U": "Topo",
    "R": "Direita",
    "F": "Frente",
    "D": "Baixo",
    "L": "Esquerda",
    "B": "Tras",
}

def carregar_faces(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Erro: arquivo nao encontrado: {caminho}")

def validar_faces(faces):
    faltando = [face for face in FACE_ORDEM if face not in faces]
    if faltando:
        raise SystemExit(f"Erro: faces faltando no JSON: {', '.join(faltando)}")

    for face in FACE_ORDEM:
        grade = faces[face]
        if not isinstance(grade, list) or len(grade) != 3:
            raise SystemExit(f"Erro: face {face} nao tem grade 3x3")
        for linha in grade:
            if not isinstance(linha, list) or len(linha) != 3:
                raise SystemExit(f"Erro: face {face} nao tem grade 3x3")
            for cor in linha:
                if cor == "Nenhum":
                    raise SystemExit("Erro: ha cores indefinidas (Nenhum) no JSON")

def eh_arquivo_convertido(faces):
    for face in FACE_ORDEM:
        for linha in faces[face]:
            for cor in linha:
                if cor not in FACE_ORDEM:
                    return False
    return True

def validar_convertido(faces):
    for face in FACE_ORDEM:
        for linha in faces[face]:
            for cor in linha:
                if cor not in FACE_ORDEM:
                    raise SystemExit(
                        "Erro: o arquivo convertido deve conter apenas letras "
                        "U, R, F, D, L e B"
                    )

def estado_em_string(faces_convertidas):
    letras = []
    for face in FACE_ORDEM:
        for linha in faces_convertidas[face]:
            letras.extend(linha)
    return "".join(letras)

def escolher_entrada(arg):
    if arg:
        return arg
    return "faces_cubo_ref.json"

def resolver(estado):
    return kociemba.solve(estado)

# --- NOVA FUNÇÃO DE TRADUÇÃO DE MOVIMENTOS ---
def traduzir_solucao(solucao_kociemba):
    movimentos = solucao_kociemba.split()
    print("\n" + "="*50)
    print(" INSTRUCOES PASSO A PASSO PARA RESOLVER O CUBO ")
    print("="*50)
    print("IMPORTANTE: Segure o cubo com a face FRENTE (F) virada para voce")
    print("            e a face TOPO (U) apontando para cima (para o teto).\n")
    
    for i, mov in enumerate(movimentos):
        letra = mov[0]
        modificador = mov[1:] if len(mov) > 1 else ""
        
        nome_face = FACE_NOMES.get(letra, letra)
        
        if modificador == "'":
            direcao = "90 graus no sentido ANTI-HORARIO"
        elif modificador == "2":
            direcao = "180 graus (Meia volta completa)"
        else:
            direcao = "90 graus no sentido HORARIO"
            
        print(f"Passo {i+1:02d}: Gire a face {nome_face.upper():<8} -> {direcao}")
        
    print("\nParabens! O cubo deve estar resolvido!")
    print("="*50)

def main():
    entrada = escolher_entrada(sys.argv[1] if len(sys.argv) > 1 else None)

    faces = carregar_faces(entrada)
    validar_faces(faces)
    
    if eh_arquivo_convertido(faces):
        faces_convertidas = faces
        print(f"Usando arquivo convertido: {entrada}")
    else:
        raise SystemExit(
            "Erro: o resolvedor espera faces_cubo_ref.json convertido. "
            "Rode mapear_faces.py para gerar o arquivo final."
        )

    validar_convertido(faces_convertidas)
    estado = estado_em_string(faces_convertidas)

    print("Estado do Cubo (Notacao Kociemba):")
    print(estado)

    try:
        solucao = resolver(estado)
    except ValueError as exc:
        print("Erro: estado invalido para o kociemba.")
        print("Dicas: verifique se o arquivo convertido usa a ordem U R F D L B corretamente.")
        raise SystemExit(1) from exc

    print("\nSolucao Bruta:")
    print(solucao)
    
    # Chama o tradutor para imprimir o passo a passo formatado
    traduzir_solucao(solucao)

if __name__ == "__main__":
    main()