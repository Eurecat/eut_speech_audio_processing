# Comparison: `snr20` vs `clean`

Change = `clean` minus `snr20`, in percentage points. 🟢 better, 🔴 worse.

| excerpt | metric | snr20 | clean | change |
|---|---|---|---|---|
| spa_0018_2spk | WER all | 30.0% | 28.5% | 🟢 −1.5 pts |
|  | WER long | 26.8% | 19.6% | 🟢 −7.2 pts |
|  | Speaker acc | 65.8% | 72.5% | 🟢 +6.7 pts |
|  | Speaker acc long | 85.7% | 100.0% | 🟢 +14.3 pts |
|  | DER fair (activity) | 27.1% | 35.4% | 🔴 +8.3 pts |
|  | DER strict (activity) | 39.9% | 47.5% | 🔴 +7.7 pts |
| spa_0019_4spk | WER all | 39.5% | 49.3% | 🔴 +9.8 pts |
|  | WER long | 20.6% | 39.0% | 🔴 +18.4 pts |
|  | Speaker acc | 50.9% | 43.1% | 🔴 −7.8 pts |
|  | Speaker acc long | 58.3% | 70.0% | 🟢 +11.7 pts |
|  | DER fair (activity) | 62.2% | 58.0% | 🟢 −4.2 pts |
|  | DER strict (activity) | 66.4% | 64.1% | 🟢 −2.3 pts |
| wer_es__es_es_weather_wer | WER all | — | 0.0% |  — |
|  | WER long | — | — |  — |
|  | Speaker acc | — | 90.0% |  — |
|  | Speaker acc long | — | — |  — |
|  | DER fair (activity) | — | 37.0% |  — |
|  | DER strict (activity) | — | 44.0% |  — |
| all paired (2 files) | WER all | 34.3% | 37.9% | 🔴 +3.6 pts |
|  | WER long | 24.5% | 26.9% | 🔴 +2.4 pts |
|  | Speaker acc | 57.3% | 56.0% | 🔴 −1.2 pts |
|  | Speaker acc long | 73.1% | 88.0% | 🟢 +14.9 pts |
|  | DER fair (activity) | 44.6% | 46.7% | 🔴 +2.1 pts |
|  | DER strict (activity) | 53.2% | 55.8% | 🔴 +2.7 pts |

## Speaker status per utterance (✓ ok / ✗ wrong / ? unknown / ∅ missed)

| excerpt | snr20 | clean |
|---|---|---|
| spa_0018_2spk | ✓ 27 ✗ 13 ? 1 ∅ 10 | ✓ 29 ✗ 11 ? 0 ∅ 11 |
| spa_0019_4spk | ✓ 28 ✗ 22 ? 5 ∅ 21 | ✓ 22 ✗ 24 ? 5 ∅ 25 |
| wer_es__es_es_weather_wer | — | ✓ 9 ✗ 0 ? 1 ∅ 0 |
| all paired (2 files) | ✓ 55 ✗ 35 ? 6 ∅ 31 | ✓ 51 ✗ 35 ? 5 ∅ 36 |

## spa_0018_2spk: utterance by utterance

| id | GT spk | GT text | snr20 spk | snr20 WER | clean spk | clean WER | change |
|---|---|---|---|---|---|---|---|
| u000 | B | ¿Se vais a ir el 31 vosotros? | ? unknown | 11.1% | ✓ B | 11.1% | fixed |
| u001 | A | Mira, ya veremos, ya veremos. | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u002 | B | Oye, hoy Pili ha estado diciendo aquí que el tren de... Pili Álava está aquí, ha venido la boda de Miquel. | ✓ B | 28.6% | ✓ B | 19.1% |  |
| u003 | A | ¡Ah, sí! ¡Miquel! | ✓ A | 33.3% | ✓ A | 100.0% |  |
| u004 | B | Miquel se ha casado con Nicole. | ✓ B | 16.7% | ✓ B | 16.7% |  |
| u005 | A | ¡Así! | ✗ B | 0.0% | ∅  | 100.0% |  |
| u006 | B | Colchonchon | ∅  | 100.0% | ∅  | 100.0% |  |
| u007 | A | Sí, sí, yo sé quien es Nicole. | ✗ B | 28.6% | ✓ A | 0.0% | fixed |
| u008 | B | Sí, se ha casado por lo civil porque ella es judía, ¿no? | ✓ B | 91.7% | ✓ B | 33.3% |  |
| u009 | A | Ah, no sabía. | ✗ B | 0.0% | ✓ A | 33.3% | fixed |
| u010 | B | Sí, entonces se han casado por lo civil, pero una boda estrictamente familiar, ¿no? | ✓ B | 14.3% | ✓ B | 21.4% |  |
| u011 | A | Imagínate | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u012 | B | Sí, y se casó el martes. Y entonces ha venido Begochu, su marido, y Pili. Y hoy se han ido Begochu y su marido y se ha quedado Pili hasta el domingo. | ✓ B | 9.4% | ✓ B | 9.4% |  |
| u013 | A | Está bien. | ∅  | 100.0% | ∅  | 100.0% |  |
| u014 | B | Entonces nos estaba diciendo que hay un tren de París a Vendaya. | ✓ B | 75.0% | ✓ B | 75.0% |  |
| u015 | A | En Daya. | ✓ A | 100.0% | ✗ B | 100.0% | broke |
| u016 | B | Que es una maravilla, ¿vale? | ✓ B | 0.0% | ✗ A | 0.0% | broke |
| u017 | A | No, sí, nosotros... | ✓ A | 0.0% | ∅  | 100.0% | broke |
| u018 | B | 3 o 4 horas | ✗ A | 50.0% | ✓ B | 50.0% | fixed |
| u019 | A | Sí, bueno, no, no, no tanto, no tanto. | ✓ A | 12.5% | ✓ A | 0.0% |  |
| u020 | B | Cuatro horas, ha dicho. | ✗ A | 0.0% | ✓ B | 0.0% | fixed |
| u021 | A | Bueno, ya es un poco más. Ese es el mismo tren que nosotros agarramos cuando fuimos ama. | ✓ A | 5.9% | ✓ A | 5.9% |  |
| u022 | B | Si, ¿Cuál es el Tafo eso? | ✗ A | 83.3% | ✓ B | 66.7% | fixed |
| u023 | A | El TDB | ✓ A | 50.0% | ✗ B | 50.0% | broke |
| u024 | B | ¿TDB? | ✗ A | 100.0% | ✗ A | 100.0% |  |
| u025 | A | TDB, TGV, TREN. | ✓ A | 66.7% | ✓ A | 66.7% |  |
| u026 | B | TGV | ∅  | 100.0% | ∅  | 100.0% |  |
| u027 | A | Tren de gran velocidad. | ✓ A | 0.0% | ✓ A | 50.0% |  |
| u028 | B | ¡Ah, el tac sí, sí, sí! ¡Tec, tec, ya, ya! | ∅  | 100.0% | ∅  | 100.0% |  |
| u029 | A | Tec. | ∅  | 100.0% | ∅  | 100.0% |  |
| u030 | B | Tec Tec | ✗ A | 100.0% | ✓ B | 100.0% | fixed |
| u031 | A | T-G-B | ✓ A | 0.0% | ✗ extra:speaker4 | 0.0% | broke |
| u032 | B | TGB, ajá, ya. | ✗ A | 33.3% | ✗ extra:speaker4 | 33.3% |  |
| u033 | A | El tren de gran vitez | ✓ A | 20.0% | ✓ A | 40.0% |  |
| u034 | B | ¡Gracias! | ✗ A | 100.0% | ✗ extra:speaker4 | 100.0% |  |
| u035 | A | Eso es. | ✓ A | 0.0% | ✗ extra:speaker4 | 0.0% | broke |
| u036 | B | Ya, ya, ya. ¿Y? | ✗ A | 25.0% | ✗ extra:speaker4 | 0.0% |  |
| u037 | A | No, pero ese dura como cinco horas. | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u038 | B | a sí?. | ✗ A | 100.0% | ✗ A | 100.0% |  |
| u039 | A | Sí, pero bueno, claro. | ✓ A | 0.0% | ✓ A | 75.0% |  |
| u040 | B | Pero vale la pena. | ∅  | 100.0% | ∅  | 100.0% |  |
| u041 | A | No, por supuesto que vale la pena, por supuesto que vale la pena. | ∅  | 100.0% | ✓ A | 46.2% | fixed |
| u042 | B | pero | ∅  | 100.0% | ∅  | 100.0% |  |
| u043 | A | Pero claro, hay que ver, ¿no? Hay que ver, o sea, yo por supuesto que si yo veo que los días que vamos a estar en París, porque ahora el problema que se me pudiera presentar es que del 24 al 31, | ✓ A | 4.5% | ✓ A | 4.5% |  |
| u044 | B | hmm | ∅  | 100.0% | ∅  | 100.0% |  |
| u045 | A | En París no sé cómo estará el movimiento de trabajo. | ✓ A | 0.0% | ✓ A | 10.0% |  |
| u046 | B | De apartamentos ya. | ∅  | 100.0% | ∅  | 100.0% |  |
| u047 | A | Yo supongo que un inmobiliario debe trabajar, ¿no? | ✓ A | 12.5% | ✓ A | 12.5% |  |
| u048 | B | Sí, sí, sí. | ✗ A | 0.0% | ✗ A | 0.0% |  |
| u049 | A | Así que... | ✓ A | 0.0% | ✓ A | 50.0% |  |
| u050 | B | Le voy a decir una cosa, París, ¿qué es el lugar? | ✓ B | 9.1% | ✓ B | 9.1% |  |

## spa_0019_4spk: utterance by utterance

| id | GT spk | GT text | snr20 spk | snr20 WER | clean spk | clean WER | change |
|---|---|---|---|---|---|---|---|
| u000 | B2 | Gracias por ver el video. | ∅  | 100.0% | ? unknown | 100.0% |  |
| u001 | A | Sí. | ∅  | 100.0% | ✗ A1 | 0.0% |  |
| u002 | A1 | Que hay mi hijito, yo no te pude llamar el día de tus cumpleaños, mi amor reciba una felicitación. No tenía el número del teléfono, no pude hacerlo, papito. | ✓ A1 | 24.1% | ✓ A1 | 13.8% |  |
| u003 | B2 | No importa, abuelo. | ✗ A1 | 100.0% | ✗ A1 | 100.0% |  |
| u004 | B2 | Bueno, no importa, tomo. | ✓ B2 | 0.0% | ✗ A2 | 25.0% | broke |
| u005 | A1 | Bueno, mi amor, ¿cómo que estás? ¿Bien? | ✗ B2 | 14.3% | ✗ A2 | 14.3% |  |
| u006 | B2 | sí | ✓ B2 | 0.0% | ✗ A2 | 0.0% | broke |
| u007 | A1 | ¿Y ya vas a entrar al colegio? | ✗ B2 | 0.0% | ✗ A2 | 85.7% |  |
| u008 | B2 | ¡Ya entré! ¡Uh! | ✓ B2 | 0.0% | ✗ A2 | 66.7% | broke |
| u009 | A1 | Silencio. | ∅  | 100.0% | ∅  | 100.0% |  |
| u010 | B2 | Silencio. | ∅  | 100.0% | ∅  | 100.0% |  |
| u011 | A1 | Haces primero, ¿verdad? | ∅  | 100.0% | ∅  | 100.0% |  |
| u012 | B2 | ¿Eh? | ∅  | 100.0% | ✗ A2 | 100.0% |  |
| u013 | A1 | Ajá. Y muy juicioso, papá. | ✓ A1 | 80.0% | ✓ A1 | 40.0% |  |
| u014 | B2 | Sí. | ∅  | 100.0% | ✗ A1 | 0.0% |  |
| u015 | A1 | Bueno, muy bien, mi amorcito. | ✓ A1 | 40.0% | ✓ A1 | 40.0% |  |
| u016 | A | Mira, Samuel. | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u017 | B2 | ¿Qué? | ∅  | 100.0% | ∅  | 100.0% |  |
| u018 | A | Le tienes que decir a tu mamá que te mande para acá, para donde tu tía Diana. | ✓ A | 5.9% | ∅  | 100.0% | broke |
| u019 | B2 | Bueno. | ✗ A | 0.0% | ✗ A | 0.0% |  |
| u020 | A | Oíste. | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u021 | B2 | Bueno. | ∅  | 100.0% | ∅  | 100.0% |  |
| u022 | A | con Alex. | ✓ A | 0.0% | ✓ A | 50.0% |  |
| u023 | B2 | Bueno. | ∅  | 100.0% | ∅  | 100.0% |  |
| u024 | A | ¿Ok? | ∅  | 100.0% | ∅  | 100.0% |  |
| u025 | B2 | Bueno. | ✗ A | 0.0% | ∅  | 100.0% |  |
| u026 | A | Mira cómo está el colegio. | ∅  | 100.0% | ✓ A | 100.0% | fixed |
| u027 | B2 | Bien | ✗ A | 0.0% | ✗ A | 0.0% |  |
| u028 | A | Bien, ¿en qué año que estás, mi amor? | ✓ A | 0.0% | ✓ A | 50.0% |  |
| u029 | B2 | Sexto. | ✗ A | 100.0% | ✗ A | 0.0% |  |
| u030 | A | ¡Sexto! | ✗ B2 | 0.0% | ✗ A2 | 0.0% |  |
| u031 | B2 | Sí. | ✓ B2 | 0.0% | ∅  | 100.0% | broke |
| u032 | A | iiii | ∅  | 100.0% | ∅  | 100.0% |  |
| u033 | B2 | Primero de bachillerato. | ✓ B2 | 0.0% | ✗ A2 | 33.3% | broke |
| u034 | A | Primero de bachillerato..., no mijito, cuando vaya yo para allá te vas a estar graduando ya. | ✓ A | 12.5% | ✓ A | 18.8% |  |
| u035 | B2 | Sí. | ✗ A | 100.0% | ✗ A | 100.0% |  |
| u036 | A | Y mire, ¿qué quieres estudiar? Todavía no sabes. | ✓ A | 25.0% | ✓ A | 25.0% |  |
| u037 | B2 | No, todavía no sé. | ✗ A | 25.0% | ✗ A | 25.0% |  |
| u038 | A | Todavía no sabes. | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u039 | B2 | ah. | ∅  | 100.0% | ∅  | 100.0% |  |
| u040 | A | Ay, pero chévere. Ah, mira, aquí Gabriel te quiere saludar otra vez. Espera tu montico, ¿ok? | ✓ A | 37.5% | ✓ A | 31.2% |  |
| u041 | B2 | ¿Ok? | ✗ A | 100.0% | ✗ A | 100.0% |  |
| u042 | A2 | Hola, Samuel | ✗ B2 | 0.0% | ✓ A2 | 0.0% | fixed |
| u043 | A | ¡Papapá! | ∅  | 100.0% | ∅  | 100.0% |  |
| u044 | B2 | Hola. | ∅  | 100.0% | ∅  | 100.0% |  |
| u045 | A2 | Hola, Sumoy. Hola. | ✗ B2 | 66.7% | ✓ A2 | 66.7% | fixed |
| u046 | A2 | Um, ¿yo puedo hablar con tu hermanita? | ✗ B2 | 14.3% | ✓ A2 | 0.0% | fixed |
| u047 | B2 | ¿Qué? | ✓ B2 | 0.0% | ∅  | 100.0% | broke |
| u048 | A2 | ¿Dónde está tu hermana? | ∅  | 100.0% | ∅  | 100.0% |  |
| u049 | B2 | ¿Cómo? | ∅  | 100.0% | ∅  | 100.0% |  |
| u050 | A2 | ¿Dónde está su hermana? | ✗ B2 | 100.0% | ∅  | 100.0% |  |
| u051 | B2 | Está en la casa. O sea, no está aquí. | ? unknown | 0.0% | ? unknown | 0.0% |  |
| u052 | A2 | Está aquí, está alla. | ✗ B2 | 0.0% | ✓ A2 | 0.0% | fixed |
| u053 | B2 | Sí. | ✓ B2 | 0.0% | ✗ A2 | 0.0% | broke |
| u054 | A2 | Entonces, ¿tú hablas inglés? | ∅  | 100.0% | ∅  | 100.0% |  |
| u055 | B2 | ¡Oh, Felipe! | ∅  | 100.0% | ✗ A2 | 100.0% |  |
| u056 | A2 | Ah | ∅  | 100.0% | ✓ A2 | 100.0% | fixed |
| u057 | B2 | mas o menos | ? unknown | 100.0% | ✗ A2 | 100.0% |  |
| u058 | A2 | ¿Qué sabes? | ? unknown | 100.0% | ? unknown | 50.0% |  |
| u059 | B2 | ¿Qué si se? | ✓ B2 | 100.0% | ? unknown | 0.0% | broke |
| u060 | A2 | ¿Sí sabes? | ? unknown | 100.0% | ∅  | 100.0% |  |
| u061 | B2 | Bueno, dale, sí. | ✓ B2 | 33.3% | ✗ A2 | 33.3% | broke |
| u062 | A2 | Sí, bueno, vale. | ✗ B2 | 66.7% | ✓ A2 | 66.7% | fixed |
| u063 | B2 | Dale, háblame algo. | ✓ B2 | 33.3% | ✗ A2 | 33.3% | broke |
| u064 | A2 | Si sabes? | ✗ B2 | 100.0% | ✓ A2 | 0.0% | fixed |
| u065 | B2 | ¿Qué? | ✓ B2 | 100.0% | ∅  | 100.0% | broke |
| u066 | A1 | Sí, sabe algo, papá. | ✗ B2 | 0.0% | ∅  | 100.0% |  |
| u067 | B2 | Yo no sé. | ✓ B2 | 33.3% | ✗ A1 | 33.3% | broke |
| u068 | A2 | Ah, ok. Cuando vaya a Colombia, yo te enseño, ¿ok? | ✗ B2 | 90.0% | ✓ A2 | 60.0% | fixed |
| u069 | B2 | Ok. | ✓ B2 | 100.0% | ✗ A2 | 100.0% | broke |
| u070 | A2 | ¡Chau! | ✗ B2 | 100.0% | ✓ A2 | 100.0% | fixed |
| u071 | B2 | ¿Ya? ¿Cómo? | ✓ B2 | 100.0% | ? unknown | 50.0% | broke |
| u072 | A2 | Chao. | ? unknown | 100.0% | ∅  | 100.0% |  |
| u073 | A | Bueno, mi vida. | ✓ A | 0.0% | ✓ A | 0.0% |  |
| u074 | B2 | Bueno. | ✗ A | 0.0% | ∅  | 100.0% |  |
| u075 | A | Entonces me le mando un beso a... | ✓ A | 0.0% | ∅  | 100.0% | broke |
