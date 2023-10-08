import numpy as np
from sklearn.datasets import load_digits
import copy

pattern_points = []
digits = load_digits()
digits_data = digits.data


def gaussian(size, sigma):
    gaussian = np.zeros((size, size))
    shift = int(size / 2)
    for i in range(-shift, shift + 1):
        for j in range(-shift, shift + 1):
            gaussian[i + shift, j + shift] = 1 / (2 * np.pi * sigma ** 2) * np.exp(
                -(i ** 2 + j ** 2) / (2 * sigma ** 2))
    return gaussian / np.sum(gaussian), shift


def fast(image_halftone, t):
    special_dots = np.zeros(shape=(len(image_halftone), len(image_halftone[0])))
    keys_arr = []
    for i in range(len(special_dots) - 6):
        for j in range(len(special_dots[0]) - 6):
            current_dot = image_halftone[i + 3, j + 3]
            circle_dots = brezenham_circle(i + 3, j + 3, image_halftone)
            if not ((image_halftone[i, j + 3] > current_dot + t and image_halftone[i + 6, j + 3] > current_dot + t) or (
                    image_halftone[i, j + 3] < current_dot - t and image_halftone[i + 6, j + 3] < current_dot - t)):
                if not ((image_halftone[i + 3, j] > current_dot + t and image_halftone[
                    i + 3, j + 6] > current_dot + t) or (image_halftone[i + 3, j] < current_dot - t and image_halftone[
                    i + 3, j + 6] < current_dot - t)):
                    continue
            for dot in range(len(circle_dots[2])):
                is_lower = True
                for c in range(1, 13):
                    if c == 1:
                        if current_dot + t < circle_dots[dot + c][2]:
                            is_lower = False
                        continue
                    if c == 12:
                        if (not is_lower and current_dot + t < circle_dots[dot + c][2]) or (
                                is_lower and current_dot + t > circle_dots[dot + c][2]):
                            special_dots[i + 3, j + 3] = 255
                            keys_arr.append([i + 3, j + 3])
                    if not is_lower and current_dot + t < circle_dots[dot + c][2] and c != 12:
                        continue
                    elif is_lower and current_dot + t > circle_dots[dot + c][2] and c != 12:
                        continue
                    else:
                        break
    return special_dots, keys_arr


def brezenham_circle(current_dot_i, current_dot_j, image_halftone):
    circle_dots = [[current_dot_i - 3, current_dot_j, image_halftone[current_dot_i - 3, current_dot_j]],  # 1
                   [current_dot_i - 3, current_dot_j + 1, image_halftone[current_dot_i - 3, current_dot_j + 1]],  # 2
                   [current_dot_i - 2, current_dot_j + 2, image_halftone[current_dot_i - 2, current_dot_j + 2]],  # 3
                   [current_dot_i - 1, current_dot_j + 3, image_halftone[current_dot_i - 1, current_dot_j + 3]],  # 4
                   [current_dot_i, current_dot_j + 3, image_halftone[current_dot_i, current_dot_j + 3]],  # 5
                   [current_dot_i + 1, current_dot_j + 3, image_halftone[current_dot_i + 1, current_dot_j + 3]],  # 6
                   [current_dot_i + 2, current_dot_j + 2, image_halftone[current_dot_i + 2, current_dot_j + 2]],  # 7
                   [current_dot_i + 3, current_dot_j + 1, image_halftone[current_dot_i + 3, current_dot_j + 1]],  # 8
                   [current_dot_i + 3, current_dot_j, image_halftone[current_dot_i + 3, current_dot_j]],  # 9
                   [current_dot_i + 3, current_dot_j - 1, image_halftone[current_dot_i + 3, current_dot_j - 1]],  # 10
                   [current_dot_i + 2, current_dot_j - 2, image_halftone[current_dot_i + 2, current_dot_j - 2]],  # 11
                   [current_dot_i + 1, current_dot_j - 3, image_halftone[current_dot_i + 1, current_dot_j - 3]],  # 12
                   [current_dot_i, current_dot_j - 3, image_halftone[current_dot_i, current_dot_j - 3]],  # 13
                   [current_dot_i - 1, current_dot_j - 3, image_halftone[current_dot_i - 1, current_dot_j - 3]],  # 14
                   [current_dot_i - 2, current_dot_j - 2, image_halftone[current_dot_i - 2, current_dot_j - 2]],  # 15
                   [current_dot_i - 3, current_dot_j - 1, image_halftone[current_dot_i - 3, current_dot_j - 1]]]  # 16
    return circle_dots


def harris_criterion(in_mtrx, keys_arr, sigma, k, N):
    R_arr = {}
    sobel_gy, sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # проход по всем точкам из FAST
    for p in keys_arr:
        M = np.zeros(shape=(2, 2))
        # проход по точкам из окна 5х5
        gaussn = gaussian(5, sigma)[0]
        for i in range(-2, 3):
            for j in range(-2, 3):
                _x, _y = p[0] + i, p[1] + j
                # подсчет частных производных в этой точке
                Ix, Iy = 0, 0
                neighborhood = in_mtrx[_x - 1: _x + 2, _y - 1: _y + 2]
                for u in range(len(sobel_gx)):
                    for v in range(len(sobel_gx[u])):
                        Ix += sobel_gx[u][v] * neighborhood[u][v]
                        Iy += sobel_gy[u][v] * neighborhood[u][v]
                # вычисление M
                A = np.array([[Ix ** 2, Ix * Iy], [Ix * Iy, Iy ** 2]])
                M = M + gaussn[i + 2, j + 2] * A
        # подсчет R
        l1, l2 = np.linalg.eigvals(M)[0], np.linalg.eigvals(M)[1]
        det_M, trace_M = l1 * l2, l1 + l2
        R = det_M - k * (trace_M ** 2)
        if R > 0:
            R_arr[R] = [p[0], p[1]]
    out = dict(sorted(R_arr.items(), reverse=True, key=lambda x: x[0]))
    return list(out.values())[:N]


def rotate_mtrx(teta):
    return np.array([[np.cos(teta), np.sin(teta)], [-np.sin(teta), np.cos(teta)]])


def brief(in_mtrx, keys_arr, patch_size, n, pattern, is_for_test):
    # добавляем нулевую рамку размером shift к матрице изображению
    true_descr = []
    all_descr = np.zeros(shape=(len(keys_arr), 30, n))
    shift = int(patch_size / 2)
    w_borders = np.pad(in_mtrx, (shift, shift))
    # проходимся по всем ключевым точкам
    all_angles = []
    true_angle = []
    angles = np.zeros(30)
    for i in range(len(angles)):
        if i == 0:
            angles[i] = 0
        else:
            angles[i] = angles[i - 1] + 2 * np.pi / 30
    # angl = 0
    # while angl < 2 * np.pi:
    #     all_angles.append(angl)
    #     angl += 2 * np.pi / 30
    for k in range(len(keys_arr)):
        x_c, y_c = keys_arr[k][0] + shift, keys_arr[k][1] + shift
        area = [[0, 0]]
        # рассматриваем область радиуса shift
        for r in range(1, shift + 1):
            for _x in range(x_c - r, x_c + r + 1):
                if r ** 2 - (_x - x_c) ** 2 < 0:
                    continue
                _y1 = int(np.round(np.sqrt(r ** 2 - (_x - x_c) ** 2) + y_c))
                _y2 = int(np.round(-np.sqrt(r ** 2 - (_x - x_c) ** 2) + y_c))
                if 0 <= _y1 < w_borders.shape[1]:
                    area.append([_x - x_c, _y1 - y_c])
                if 0 <= _y2 < w_borders.shape[1]:
                    area.append([_x - x_c, _y2 - y_c])
            for _y in range(y_c - r, y_c + r + 1):
                if r ** 2 - (_y - y_c) ** 2 < 0:
                    continue
                _x1 = int(np.round(np.sqrt(r ** 2 - (_y - y_c) ** 2) + x_c))
                _x2 = int(np.round(-np.sqrt(r ** 2 - (_y - y_c) ** 2) + x_c))
                if 0 <= _x1 < w_borders.shape[0]:
                    area.append([_x1 - x_c, _y - y_c])
                if 0 <= _x2 < w_borders.shape[0]:
                    area.append([_x2 - x_c, _y - y_c])
        area = np.unique(area, axis=0)

        # генерируем n пар точек из области и заполняем матрицу S
        S = np.zeros(shape=(2, n, 2))
        if k == 0 and not is_for_test:
            for i in range(n):
                while True:
                    # принадлежат area, не принадлежат pattern_points чтобы были пары точек не повторялись,
                    rand = np.random.normal(0, patch_size ** 2 / 25, 4).astype(int)
                    if [rand[0], rand[1]] in area.tolist() and [rand[2], rand[3]] in area.tolist():
                        if not [rand[0], rand[1]] in pattern_points and not [rand[2], rand[3]] in pattern_points:
                            pattern_points.append([rand[0], rand[1]])
                            pattern_points.append([rand[2], rand[3]])
                            break
        for i in range(n):
            u1, u2 = pattern_points[2 * i], pattern_points[2 * i + 1]
            S[0, i] = [u1[0], u1[1]]
            S[1, i] = [u2[0], u2[1]]
        minimum = 36000
        angle = 0
        for a in angles:
            dif = np.abs(a - keys_arr[k][2])
            if dif < minimum:
                minimum = dif
                angle = a
        teta = angle
        for a in range(len(angles)):
            _S1, _S2 = S[0].T, S[1].T
            S1 = (rotate_mtrx(angles[a]) @ _S1).astype(int)
            S2 = (rotate_mtrx(angles[a]) @ _S2).astype(int)
            bin_row = np.zeros(shape=n)
            for i in range(n):
                p1, p2 = S1.T[i], S2.T[i]
                if w_borders[x_c + p1[0], y_c + p1[1]] < w_borders[x_c + p2[0], y_c + p2[1]]:
                    bin_row[i] = 1
            if teta == angles[a]:
                true_descr.append(list(bin_row))
                all_descr[k, a] = bin_row
            else:
                all_descr[k, a] = bin_row
    if not is_for_test:
        return true_descr, all_descr, true_angle, all_angles, pattern
    else:
        return true_descr, all_descr, true_angle, all_angles


def histogram(digits, n):
    pix_counts = np.zeros((len(digits), n))
    for i in range(len(digits)):
        for j in range(len(digits[0])):
            ind = int(digits[i][j])
            pix_counts[i][ind] += 1
    return pix_counts


def data_distribution(array, centroid, cluster_target, target):
    k = len(centroid)
    cluster_content = [[] for i in range(k)]
    n = len(array)
    param = len(array[0])
    for i in range(n):
        t = target[i]
        min_distance = 999999
        suitable_cluster = -1
        for j in range(k):
            distance = 0
            for p in range(param):
                distance += np.sqrt((array[i][p] - centroid[j][p]) ** 2)
            if distance < min_distance:
                min_distance = distance
                suitable_cluster = j
        cluster_content[suitable_cluster].append(array[i])
        cluster_target[suitable_cluster].append(t)
    return cluster_content, cluster_target


def centroid_update(centroid, cluster_content, param):
    k = len(centroid)
    for i in range(k): #по i кластерам
        for p in range(param): #по q параметрам
            updated_parameter = 0
            for j in range(len(cluster_content[i])):
                updated_parameter += cluster_content[i][j][p]
            if len(cluster_content[i]) != 0:
                updated_parameter = updated_parameter / len(cluster_content[i])
            centroid[i][p] = updated_parameter
    return centroid


def clusterization(array, k, target):
    param = len(array[0])
    centroid = [[0 for i in range(param)] for q in range(k)]
    cluster_target = [[] for q in range(k)]

    # Рандомно из выборки
    for q in range(k):
        rand_ind = np.random.randint(0, len(array))
        centroid[q] = list(array[rand_ind].astype(int))

    # Дальнее расстояние
    # for q in range(k):
    #     if q % 2 == 0:
    #         rand_ind = np.random.randint(0, len(array))
    #         centroid[q] = list(array[rand_ind].astype(int))
    #     else:
    #         max_distance = 0
    #         ind = -1
    #         for i in range(len(array)):
    #             distance = 0
    #             for p in range(param):
    #                 distance += np.sqrt((centroid[q-1][p] - array[i][p]) ** 2)
    #             if distance > max_distance:
    #                 already_added = False
    #                 for c in range(len(centroid)):
    #                     if centroid[c] == list(array[i].astype(int)):
    #                         already_added = True
    #                 if not already_added:
    #                     max_distance = distance
    #                     ind = i
    #         centroid[q] = list(array[ind].astype(int))

    # Рандомные параметры
    # for p in range(param):
    #     for q in range(k):
    #         centroid[q][p] = np.random.randint(0, 16)

    cluster_content, cluster_target = data_distribution(array, centroid, cluster_target, target)
    prev_centroid = copy.deepcopy(centroid)
    while 1:
        centroid = centroid_update(centroid, cluster_content, param)
        cluster_content, cluster_target = data_distribution(array, centroid, cluster_target, target)
        if centroid == prev_centroid:
            break
        prev_centroid = copy.deepcopy(centroid)
    predicted = np.zeros(shape=len(array))
    cluster_t = [[] for q in range(k)]
    for i in range(len(cluster_t)):
        while 1:
            if np.argmax(np.bincount(cluster_target[i])) not in cluster_t:
                cluster_t[i] = np.argmax(np.bincount(cluster_target[i]))
                break
            else:
                cluster_target[i] = np.delete(cluster_target[i], np.where(cluster_target[i] == np.argmax(np.bincount(cluster_target[i]))))
    for i in range(len(array)):
        for c in range(len(cluster_content)):
            for k in range(len(cluster_content[c])):
                if np.array_equal(array[i], cluster_content[c][k]):
                    predicted_target = cluster_t[c]
                    predicted[i] = predicted_target
    return cluster_content, centroid, predicted.astype(int), cluster_t


# # среднее внутрикластерное расстояние
# def intra_cluster_distance(cluster_content):
#     intra_cluster_distance = 0
#     distance_in_one_cluster = []
#     for k in range(len(cluster_content)):
#         distance = 0
#         for i in range(len(cluster_content[k])):
#             for j in range(i+1, len(cluster_content[k])-1):
#                 param = len(cluster_content[k][i])
#                 for p in range(param):
#                     distance += np.sqrt((cluster_content[k][i][p] - cluster_content[k][j][p]) ** 2)
#         distance_in_one_cluster.append(distance)
#         if len(cluster_content[k]) != 0:
#             intra_cluster_distance += distance / len(cluster_content[k])
#     return intra_cluster_distance
#
#
# # среднее межкластерное расстояние
# def inter_cluster_distance(cluster_content):
#     inter_cluster_distance = 0
#     distance_between_2_clusters = []
#     for k in range(len(cluster_content)):   # выбранный кластер
#         distance = 0
#         len_k2 = 0
#         mean_distance_between_2_clusters = 0
#         for i in range(len(cluster_content[k])):    # выбранная точка
#             param = len(cluster_content[k][i])
#             for k2 in range(k+1, len(cluster_content)-1):   # все остальные кластеры
#                 if i == 0:
#                     len_k2 += len(cluster_content[k2])  # размер кластера
#                 for i2 in range(len(cluster_content[k2])):  # по точкам остальных кластеров
#                     for p in range(param):
#                         distance += np.sqrt((cluster_content[k][i][p] - cluster_content[k2][i2][p]) ** 2)
#         distance_between_2_clusters.append(distance)
#         if len(cluster_content[k]) != 0:
#             inter_cluster_distance += distance / (len(cluster_content[k])+len_k2)
#     return inter_cluster_distance


def digits_arr_to_mtrx(digits):
    digits_mtrxs = np.zeros(shape=(len(digits), int(np.sqrt(len(digits[0]))), int(np.sqrt(len(digits[0])))))
    for d in range(len(digits)):
        k = 0
        for i in range(int(np.sqrt(len(digits[0])))):
            for j in range(int(np.sqrt(len(digits[0])))):
                digits_mtrxs[d][i][j] = digits[d][k]
                k += 1
    return digits_mtrxs


def local_binary_pattern(digits_mtrxs):
    binary_arr = np.zeros(shape=(len(digits_mtrxs), (len(digits_mtrxs[0]))*(len(digits_mtrxs[0][0])), 8))
    lbp = np.zeros(shape=(len(digits_mtrxs), (len(digits_mtrxs[0]))*(len(digits_mtrxs[0][0]))))
    for d in range(len(digits_mtrxs)):
        iterate = 0
        digits_mtrx = np.copy(digits_mtrxs[d])
        digits_mtrx = np.concatenate((np.zeros((1, len(digits_mtrx[0]))), digits_mtrx, np.zeros((1, len(digits_mtrx[0])))),
                                      axis=0)
        digits_mtrx = np.concatenate((np.zeros((len(digits_mtrx[0]) + 2, 1)), digits_mtrx, np.zeros((len(digits_mtrx[0]) + 2, 1))),
                                      axis=1)
        for i in range(1, len(digits_mtrx)-1):
            for j in range(1, len(digits_mtrx[0])-1):
                pix = digits_mtrx[i][j]
                neighborhood = digits_mtrx[i - 1: i + 2, j - 1: j + 2]
                if pix > neighborhood[0][0]: binary_arr[d][iterate][7] = 1
                if pix > neighborhood[0][1]: binary_arr[d][iterate][6] = 1
                if pix > neighborhood[0][2]: binary_arr[d][iterate][5] = 1
                if pix > neighborhood[1][2]: binary_arr[d][iterate][4] = 1
                if pix > neighborhood[2][2]: binary_arr[d][iterate][3] = 1
                if pix > neighborhood[2][1]: binary_arr[d][iterate][2] = 1
                if pix > neighborhood[2][0]: binary_arr[d][iterate][1] = 1
                if pix > neighborhood[1][0]: binary_arr[d][iterate][0] = 1
                iterate += 1
    for b in range(len(binary_arr)):
        for i in range(len(binary_arr[0])):
            strng = ""
            for j in range(len(binary_arr[0][0])):
                strng += str(int(binary_arr[b][i][j]))
            lbp[b][i] = int(strng, 2)
    pix_counts = np.zeros(shape=(len(lbp), int(np.max(lbp) + 1)))
    for _i in range(len(lbp)):
        for _j in range(len(lbp[0])):
            ind = int(lbp[_i][_j])
            pix_counts[_i][ind] += 1
    return lbp, pix_counts


target = digits.target
target_names = digits.target_names
digits_mtrx = digits_arr_to_mtrx(digits_data)
lbp, lbp_hist = local_binary_pattern(digits_mtrx)
luster_content, centroid, predict, cluster_t = clusterization(lbp_hist, 10, target)
print("local binary pattern метод")
# print("centroids after k_means: ")
# for i in range(len(centroid)):
#     print(centroid[i])
print(f'predict: {predict}')
print(f'target: {digits.target}')
success = np.zeros(shape=len(predict))
for i in range(len(predict)):
    if predict[i] == target[i]:
        success[i] = 1
print(f'guessed: {np.sum(success)/len(success)}')

print("\n\n гистограмный метод")
hist = histogram(digits_data, 17)
cluster_content, centroid, predict, cluster_t = clusterization(hist, 10, target)
# print("centroids after k_means: ")
# for i in range(len(centroid)):
#     print(centroid[i])
print(f'predict: {predict}')
print(f'target: {digits.target}')
success = np.zeros(shape=len(predict))
for i in range(len(predict)):
    if predict[i] == target[i]:
        success[i] = 1
print(f'guessed: {np.sum(success)/len(success)}')


# Кластеризацию провести методом k-means на основе анализа гистограммы, а также на основе характеристик, полученных
# на основе получения угловых точек Харриса и их дескрипторов (по 4 точки). Если такого количества нет,
# соответствующие характеристики считать нулевыми.
#
# Оценить качество кластеризации с помощью известных меток классов изображений.
