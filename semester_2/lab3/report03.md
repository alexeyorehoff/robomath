# Лабораторная работа 3 — Формула Байеса


## 1) Формула Байеса

### Условие

В Гонконге все такси либо синие, либо зеленые. Свидетель утверждает, что видел синее такси. Вероятность правильного распознавания цвета в условиях плохой видимости — 75%.

---

### (a) Дано:

- P(Синий) = 0.1, P(Зелёный) = 0.9
- P(Свидетель говорит "Синий" | Синий) = 0.75
- P(Свидетель говорит "Синий" | Зелёный) = 0.25

#### По формуле Байеса:

$$
P(Синий | Свидетель\ говорит\ "Синий") = \frac{P("Синий" | Синий) \cdot P(Синий)}{P("Синий")}
$$

$$
P("Синий") = 0.75 \cdot 0.1 + 0.25 \cdot 0.9 = 0.075 + 0.225 = 0.3
$$

$$
P(Синий | "Синий") = \frac{0.75 \cdot 0.1}{0.3} = \frac{0.075}{0.3} = 0.25
$$

#### **Ответ:** 25%

---

### (b) Дано:

- P(Синий) = 0.3, P(Зелёный) = 0.7

$$
P("Синий") = 0.75 \cdot 0.3 + 0.25 \cdot 0.7 = 0.225 + 0.175 = 0.4
$$

$$
P(Синий | "Синий") = \frac{0.75 \cdot 0.3}{0.4} = \frac{0.225}{0.4} = 0.5625
$$

#### **Ответ:** 56.25%

---

### (c) Второй свидетель — дальтоник (точность 50%)

#### Учитываем двоих свидетелей:

- Первый: вероятность синего после него — $P_1 = 0.5625$
- Второй: говорит "зелёный", но точность = 0.5

Обновим байесовскую вероятность, принимая первую как априорную:

$$
P(Синий | Свид1, Свид2) = \frac{P("Зелёный"|Синий) \cdot P_1}{P("Зелёный")}
$$

- P("Зелёный"|Синий) = 0.5, \( P("Зелёный"|Зелёный) = 0.5
- P(Зелёный) = 1 - $P_1$ = 0.4375
- P("Зелёный") = $0.5 \cdot 0.5625 + 0.5 \cdot 0.4375$ = 0.5

$$
P(Синий | Свид1, Свид2) = \frac{0.5 \cdot 0.5625}{0.5} = 0.5625
$$

#### **Ответ:** не изменилось, 56.25% (второй свидетель не внёс информации)

---

## 2) Байесовский фильтр

### Условие

- До уборки: ничего не известно → равномерное распределение: \( P(грязный) = P(чистый) = 0.5 \)
- После уборки: \( P(чистый | грязный, очистка) = 0.7 \)
- Датчик: \( P(z=чистый | грязный) = 0.3 \), \( P(z=чистый | чистый) = 0.9 \)
- Датчик показывает: "чисто"

---

### (a) Вероятность, что пол остался грязным

Сначала определим апостериорные вероятности состояния после уборки:

- \( P(x = чистый) = 0.5 \cdot 0.7 + 0.5 \cdot 0 = 0.35 \)
- \( P(x = грязный) = 0.5 \cdot 0.3 + 0.5 \cdot 1 = 0.65 \)

Теперь по формуле Байеса:

$$
P(грязный | z=чистый) = \frac{P(z=чистый | грязный) \cdot P(грязный)}{P(z=чистый)}
$$

$$
P(z=чистый) = 0.3 \cdot 0.65 + 0.9 \cdot 0.35 = 0.195 + 0.315 = 0.51
$$

$$
P(грязный | z=чистый) = \frac{0.3 \cdot 0.65}{0.51} \approx 0.382
$$

#### **Ответ:** приблизительно 38.2%

---

### (b) Нижняя граница вероятности

Чтобы оценить нижнюю границу, предполагаем худший случай:

- \( P(грязный) = 1 \), тогда:

$$
P(z=чистый | грязный) = 0.3
$$

#### **Нижняя граница:** 30%
