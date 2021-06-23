
def transpose(m):
    rows = len(m)
    columns = len(m[0])

    mt = []
    for j in range(columns):
        mt.append([])
        for i in range(rows):
            mt[j].append(m[i][j])

    return mt


def multiply(a, b):
    result = []

    for i in range(len(a)):
        result.append([])
        for j in range(len(b[0])):
            result[i].append(0)
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]

    return result


def inverse(m):
    det = determinant(m)

    if len(m) == 2:
        return [[m[1][1]/det, -1*m[0][1]/det],
                [-1*m[1][0]/det, m[0][0]/det]]

    m_inverse = []
    for j in range(len(m)):
        row = []

        for i in range(len(m)):
            submatrix = [row[:i] + row[i+1:] for row in (m[:j] + m[j+1:])]
            row.append((pow(-1, j + i) * determinant(submatrix)) / det)
        m_inverse.append(row)

    m_inverse = transpose(m_inverse)

    return m_inverse


def determinant(m):
    indices = list(range(len(m)))
    val = 0

    if len(m) == 2 and len(m[0]) == 2:
        val = m[0][0] * m[1][1] - m[1][0] * m[0][1]
        return val

    for j in indices:
        submatrix = m.copy()
        submatrix = submatrix[1:]

        for i in range(len(submatrix)):
            submatrix[i] = submatrix[i][:j] + submatrix[i][j + 1:]

        sign = (-1) ** (j % 2)

        sub_det = determinant(submatrix)
        val += sign * m[0][j] * sub_det

    return val

