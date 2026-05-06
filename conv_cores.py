import json
import sys

FACE_ORDEM = ["U", "R", "F", "D", "L", "B"]
ADJACENCIAS = {
	"U": {"top": "B", "right": "R", "bottom": "F", "left": "L"},
	"R": {"top": "U", "right": "B", "bottom": "D", "left": "F"},
	"F": {"top": "U", "right": "R", "bottom": "D", "left": "L"},
	"D": {"top": "F", "right": "R", "bottom": "B", "left": "L"},
	"L": {"top": "U", "right": "F", "bottom": "D", "left": "B"},
	"B": {"top": "U", "right": "L", "bottom": "D", "left": "R"},
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


def mapa_centros(faces):
	cor_para_face = {}
	for face in FACE_ORDEM:
		cor_centro = faces[face][1][1]
		if cor_centro in cor_para_face:
			raise SystemExit(
				"Erro: cor de centro repetida: "
				f"{cor_centro} nas faces {cor_para_face[cor_centro]} e {face}"
			)
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
				if cor not in cor_para_face:
					raise SystemExit(f"Erro: cor sem referencia de centro: {cor}")
				nova_linha.append(cor_para_face[cor])
			nova_grade.append(nova_linha)
		convertido[face] = nova_grade
	return convertido


def estado_em_string(faces_convertidas):
	letras = []
	for face in FACE_ORDEM:
		for linha in faces_convertidas[face]:
			letras.extend(linha)
	return "".join(letras)


def rotacionar_face_cw(grade):
	return [list(linha) for linha in zip(*grade[::-1])]


def orientar_face(face, grade):
	esperado = ADJACENCIAS[face]
	for rotacao in range(4):
		if (
			grade[0][1] == esperado["top"]
			and grade[1][2] == esperado["right"]
			and grade[2][1] == esperado["bottom"]
			and grade[1][0] == esperado["left"]
		):
			return grade, rotacao
		grade = rotacionar_face_cw(grade)
	return None, None


def orientar_faces(faces_convertidas):
	orientadas = {}
	rotacoes = {}
	problemas = []
	for face in FACE_ORDEM:
		grade = [linha[:] for linha in faces_convertidas[face]]
		grade_orientada, rotacao = orientar_face(face, grade)
		if grade_orientada is None:
			problemas.append(face)
			orientadas[face] = faces_convertidas[face]
		else:
			orientadas[face] = grade_orientada
			rotacoes[face] = rotacao
	return orientadas, rotacoes, problemas


def salvar_json(caminho, dados):
	with open(caminho, "w", encoding="utf-8") as f:
		json.dump(dados, f, indent=4)


def main():
	entrada = sys.argv[1] if len(sys.argv) > 1 else "faces_cubo.json"
	saida = sys.argv[2] if len(sys.argv) > 2 else "faces_cubo_ref.json"

	faces = carregar_faces(entrada)
	validar_faces(faces)

	cor_para_face = mapa_centros(faces)
	faces_convertidas = converter_faces(faces, cor_para_face)
	faces_orientadas, rotacoes, problemas = orientar_faces(faces_convertidas)
	if problemas:
		print("Erro: orientacao inconsistente nas faces: " + ", ".join(problemas))
		raise SystemExit(1)
	for face, rotacao in rotacoes.items():
		if rotacao:
			print(f"Face {face} rotacionada {rotacao * 90} graus")

	faces_convertidas = faces_orientadas

	salvar_json(saida, faces_convertidas)

	print("Mapa de centros (cor -> face):")
	for cor, face in cor_para_face.items():
		print(f"  {cor} -> {face}")

	print(f"Arquivo salvo: {saida}")
	print("Estado em string (ordem U R F D L B):")
	print(estado_em_string(faces_convertidas))


if __name__ == "__main__":
	main()
