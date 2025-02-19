
movement_deltas = []
total_distance = 0
animal_positions = [('hare', (0, 17)), ('hare', (4, 22)), ('stag', (6, 3)), ('stag1', (7, 8))]
position = (5, 5)
previous_position = (0, 0)

# Função para calcular a distância de Manhattan entre duas posições
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

for animal_id, animal_position in animal_positions:
    dist_t1 = manhattan_distance(position, animal_position)
    dist_t0 = manhattan_distance(previous_position, animal_position)
    movement_deltas.append((animal_id, dist_t1 - dist_t0))  # Armazenando como tupla
    total_distance += dist_t1

print(movement_deltas)
