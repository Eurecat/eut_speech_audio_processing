# Test: docker-compose_mp3 pipeline (2026-09-23)

Hypothesis source: file:callhome_spa_snr20__spa_0018_2spk_snr20.wav, file:callhome_spa_snr20__spa_0019_4spk_snr20.wav, file:wer_es__es_es_weather_wer.wav  
WER long = only GT utterances with more than 6 normalised words. Speaker: pipeline ids mapped to GT labels by Hungarian matching; accuracy excludes missed utterances. DER fair/strict match `~/aimara-bench/benchmarks/scoring/metrics.py` (fair: collar 0.25s, overlap skipped — comparable to published CALLHOME/AMI numbers; strict: collar 0s, overlap scored — misses the overlapping talker, the failure that makes an agent interrupt).

| file | GT verified | utt GT / ASR | WER all | WER long (utt) | speaker ok / wrong / unknown / missed | speaker acc | speaker acc long | DER fair | DER strict |
|---|---|---|---|---|---|---|---|---|---|
| wer_es__es_es_weather_wer | 0/10 | 10 / 10 | 0.0% | — (0) | 9 / 0 / 1 / 0 | 90.0% | — | 37.0% | 44.0% |
| callhome_spa_snr20__spa_0019_4spk_snr20 | 0/76 | 76 / 47 | 39.5% | 20.6% (12) | 28 / 22 / 5 / 21 | 50.9% | 58.3% | 62.2% | 66.4% |
| callhome_spa_snr20__spa_0018_2spk_snr20 | 0/51 | 51 / 31 | 30.0% | 26.8% (16) | 27 / 13 / 1 / 10 | 65.8% | 85.7% | 27.1% | 39.9% |
| **all (pooled)** | | 137 / 88 | **31.5%** | **24.5%** (28) | 64 / 35 / 7 / 31 | **60.4%** | 73.1% | **43.6%** | **52.0%** |

## wer_es__es_es_weather_wer

Speaker mapping: `{"speaker1": "A", "speaker2": "B", "speaker3": "C"}`

| id | time | long | GT spk | pred spk | speaker | WER | GT text | ASR text |
|---|---|---|---|---|---|---|---|---|
| u000 | 0.3-3.1 |  | A | A (speaker1) | ✅ | 0.0% | Hay 17 grados con sol. | hay diecisiete grados con sol |
| u001 | 4.4-7.0 |  | B | B (speaker2) | ✅ | 0.0% | Hay 14 grados y está nublado. | hay catorce grados y esta nublado |
| u002 | 8.0-10.1 |  | B | B (speaker2) | ✅ | 0.0% | Hay 15 grados y llueve. | hay quince grados y llueve |
| u003 | 10.9-13.1 |  | C | unknown (unknown) | ❔ unknown | 0.0% | Hay 21 grados con sol. | hay veintiun grados con sol |
| u004 | 14.1-16.1 |  | C | C (speaker3) | ✅ | 0.0% | Hay 11 grados con sol. | hay once grados con sol |
| u005 | 17.1-19.3 |  | B | B (speaker2) | ✅ | 0.0% | Hay quince grados con sol | hay quince grados con sol |
| u006 | 20.3-23.3 |  | A | A (speaker1) | ✅ | 0.0% | Hay 19 grados y está nublado. | hay diecinueve grados y esta nublado |
| u007 | 25.0-27.0 |  | C | C (speaker3) | ✅ | 0.0% | Hay 20 grados con sol. | hay veinte grados con sol |
| u008 | 28.1-30.6 |  | B | B (speaker2) | ✅ | 0.0% | Hay 11 grados y está nublado. | hay once grados y esta nublado |
| u009 | 31.6-34.6 |  | A | A (speaker1) | ✅ | 0.0% | Hay 18 grados y está nublado. | hay dieciocho grados y esta nublado |

## callhome_spa_snr20__spa_0019_4spk_snr20

Speaker mapping: `{"speaker4": "A", "speaker1": "A1", "speaker2": "B2"}`

| id | time | long | GT spk | pred spk | speaker | WER | GT text | ASR text |
|---|---|---|---|---|---|---|---|---|
| u000 | 0.0-0.9 |  | B2 |  | ∅ missed | 100.0% | Gracias por ver el video. |  |
| u001 | 1.1-1.9 |  | A |  | ∅ missed | 100.0% | Sí. |  |
| u002 | 2.0-10.6 | ✔ | A1 | A1 (speaker1) | ✅ | 24.1% | Que hay mi hijito, yo no te pude llamar el día de tus cumpleaños, mi amor reciba una felicitación. No tenía el número del teléfono, no pude hacerlo, papito. | so new see yo no te pude llamar el dia de tus cumpleanos mi amor reciba una felicitacion no importa no tenia el numero del telefono no pude hacerlo |
| u003 | 6.5-8.2 |  | B2 | A1 (speaker1) | ❌ wrong | 100.0% | No importa, abuelo. | papi |
| u004 | 10.6-12.4 |  | B2 | B2 (speaker2) | ✅ | 0.0% | Bueno, no importa, tomo. | bueno no importa tomo |
| u005 | 13.0-14.9 | ✔ | A1 | B2 (speaker2) | ❌ wrong | 14.3% | Bueno, mi amor, ¿cómo que estás? ¿Bien? | bueno mi amor como estas bien |
| u006 | 13.8-15.7 |  | B2 | B2 (speaker2) | ✅ | 0.0% | sí | si |
| u007 | 15.7-17.4 | ✔ | A1 | B2 (speaker2) | ❌ wrong | 0.0% | ¿Y ya vas a entrar al colegio? | y ya vas a entrar al colegio |
| u008 | 17.2-19.2 |  | B2 | B2 (speaker2) | ✅ | 0.0% | ¡Ya entré! ¡Uh! | ya entre uh |
| u009 | 20.4-20.8 |  | A1 |  | ∅ missed | 100.0% | Silencio. |  |
| u010 | 20.5-21.2 |  | B2 |  | ∅ missed | 100.0% | Silencio. |  |
| u011 | 20.8-22.1 |  | A1 |  | ∅ missed | 100.0% | Haces primero, ¿verdad? |  |
| u012 | 22.0-22.4 |  | B2 |  | ∅ missed | 100.0% | ¿Eh? |  |
| u013 | 22.8-25.1 |  | A1 | A1 (speaker1) | ✅ | 80.0% | Ajá. Y muy juicioso, papá. | e voce wash o papa |
| u014 | 23.9-26.4 |  | B2 |  | ∅ missed | 100.0% | Sí. |  |
| u015 | 26.0-28.4 |  | A1 | A1 (speaker1) | ✅ | 40.0% | Bueno, muy bien, mi amorcito. | bueno muy bien |
| u016 | 26.0-28.4 |  | A | A (speaker4) | ✅ | 0.0% | Mira, Samuel. | mira samuel |
| u017 | 28.4-28.9 |  | B2 |  | ∅ missed | 100.0% | ¿Qué? |  |
| u018 | 28.7-32.4 | ✔ | A | A (speaker4) | ✅ | 5.9% | Le tienes que decir a tu mamá que te mande para acá, para donde tu tía Diana. | le tienes que decir a tu mama que te mande para aca para donde tu tia vive |
| u019 | 32.4-33.0 |  | B2 | A (speaker4) | ❌ wrong | 0.0% | Bueno. | bueno |
| u020 | 33.0-33.6 |  | A | A (speaker4) | ✅ | 0.0% | Oíste. | oiste |
| u021 | 33.7-34.3 |  | B2 |  | ∅ missed | 100.0% | Bueno. |  |
| u022 | 34.2-35.4 |  | A | A (speaker4) | ✅ | 0.0% | con Alex. | con alex |
| u023 | 35.5-35.9 |  | B2 |  | ∅ missed | 100.0% | Bueno. |  |
| u024 | 36.0-36.5 |  | A |  | ∅ missed | 100.0% | ¿Ok? |  |
| u025 | 36.7-37.1 |  | B2 | A (speaker4) | ❌ wrong | 0.0% | Bueno. | bueno |
| u026 | 37.1-38.4 |  | A |  | ∅ missed | 100.0% | Mira cómo está el colegio. |  |
| u027 | 38.5-39.3 |  | B2 | A (speaker4) | ❌ wrong | 0.0% | Bien | bien |
| u028 | 39.1-40.9 | ✔ | A | A (speaker4) | ✅ | 0.0% | Bien, ¿en qué año que estás, mi amor? | bien en que ano que estas mi amor |
| u029 | 39.7-41.8 |  | B2 | A (speaker4) | ❌ wrong | 100.0% | Sexto. | six |
| u030 | 42.1-42.8 |  | A | B2 (speaker2) | ❌ wrong | 0.0% | ¡Sexto! | sexto |
| u031 | 42.9-43.3 |  | B2 | B2 (speaker2) | ✅ | 0.0% | Sí. | si |
| u032 | 43.5-44.4 |  | A |  | ∅ missed | 100.0% | iiii |  |
| u033 | 44.1-45.3 |  | B2 | B2 (speaker2) | ✅ | 0.0% | Primero de bachillerato. | primero de bachillerato |
| u034 | 45.6-49.9 | ✔ | A | A (speaker4) | ✅ | 12.5% | Primero de bachillerato..., no mijito, cuando vaya yo para allá te vas a estar graduando ya. | primero no mijito cuando vaya yo para alla te vas a estar graduando ya |
| u035 | 50.0-50.8 |  | B2 | A (speaker4) | ❌ wrong | 100.0% | Sí. | ah |
| u036 | 50.8-53.7 | ✔ | A | A (speaker4) | ✅ | 25.0% | Y mire, ¿qué quieres estudiar? Todavía no sabes. | y mira y que quieres estudiar todavia no sabes |
| u037 | 53.8-54.8 |  | B2 | A (speaker4) | ❌ wrong | 25.0% | No, todavía no sé. | no todavia no sabes |
| u038 | 54.8-55.7 |  | A | A (speaker4) | ✅ | 0.0% | Todavía no sabes. | todavia no sabes |
| u039 | 56.2-56.5 |  | B2 |  | ∅ missed | 100.0% | ah. |  |
| u040 | 56.3-60.7 | ✔ | A | A (speaker4) | ✅ | 37.5% | Ay, pero chévere. Ah, mira, aquí Gabriel te quiere saludar otra vez. Espera tu montico, ¿ok? | ay pero chevere te sales ah mira aqui gabriel te quiere saludar otra vez esperate un bontico |
| u041 | 60.5-60.9 |  | B2 | A (speaker4) | ❌ wrong | 100.0% | ¿Ok? | okay |
| u042 | 61.0-61.6 |  | A2 | B2 (speaker2) | ❌ wrong | 0.0% | Hola, Samuel | hola samuel |
| u043 | 61.7-62.5 |  | A |  | ∅ missed | 100.0% | ¡Papapá! |  |
| u044 | 61.9-64.4 |  | B2 |  | ∅ missed | 100.0% | Hola. |  |
| u045 | 62.8-64.9 |  | A2 | B2 (speaker2) | ❌ wrong | 66.7% | Hola, Sumoy. Hola. | hola |
| u046 | 65.1-68.5 | ✔ | A2 | B2 (speaker2) | ❌ wrong | 14.3% | Um, ¿yo puedo hablar con tu hermanita? | come yo puedo hablar con tu hermanita |
| u047 | 68.2-69.4 |  | B2 | B2 (speaker2) | ✅ | 0.0% | ¿Qué? | que |
| u048 | 70.2-71.5 |  | A2 |  | ∅ missed | 100.0% | ¿Dónde está tu hermana? |  |
| u049 | 72.3-72.9 |  | B2 |  | ∅ missed | 100.0% | ¿Cómo? |  |
| u050 | 73.0-74.5 |  | A2 | B2 (speaker2) | ❌ wrong | 100.0% | ¿Dónde está su hermana? | que |
| u051 | 75.2-78.1 | ✔ | B2 | unknown (unknown) | ❔ unknown | 0.0% | Está en la casa. O sea, no está aquí. | esta en la casa o sea no esta aqui |
| u052 | 78.5-79.8 |  | A2 | B2 (speaker2) | ❌ wrong | 0.0% | Está aquí, está alla. | esta aqui esta alla |
| u053 | 80.1-80.5 |  | B2 | B2 (speaker2) | ✅ | 0.0% | Sí. | si |
| u054 | 80.8-85.9 |  | A2 |  | ∅ missed | 100.0% | Entonces, ¿tú hablas inglés? |  |
| u055 | 86.7-88.0 |  | B2 |  | ∅ missed | 100.0% | ¡Oh, Felipe! |  |
| u056 | 88.2-90.0 |  | A2 |  | ∅ missed | 100.0% | Ah |  |
| u057 | 89.6-90.6 |  | B2 | unknown (unknown) | ❔ unknown | 100.0% | mas o menos | i'm |
| u058 | 91.0-91.7 |  | A2 | unknown (unknown) | ❔ unknown | 100.0% | ¿Qué sabes? | doctor no |
| u059 | 92.8-93.4 |  | B2 | B2 (speaker2) | ✅ | 100.0% | ¿Qué si se? | feliz uh vo |
| u060 | 94.0-94.7 |  | A2 | unknown (unknown) | ❔ unknown | 100.0% | ¿Sí sabes? | quisise sisha |
| u061 | 95.6-98.2 |  | B2 | B2 (speaker2) | ✅ | 33.3% | Bueno, dale, sí. | bueno dale |
| u062 | 98.1-99.4 |  | A2 | B2 (speaker2) | ❌ wrong | 66.7% | Sí, bueno, vale. | si |
| u063 | 99.8-100.7 |  | B2 | B2 (speaker2) | ✅ | 33.3% | Dale, háblame algo. | dale hablame |
| u064 | 101.0-101.7 |  | A2 | B2 (speaker2) | ❌ wrong | 100.0% | Si sabes? | al |
| u065 | 102.2-102.6 |  | B2 | B2 (speaker2) | ✅ | 100.0% | ¿Qué? | tschuss |
| u066 | 102.5-103.7 |  | A1 | B2 (speaker2) | ❌ wrong | 0.0% | Sí, sabe algo, papá. | si sabe algo papa |
| u067 | 104.2-105.9 |  | B2 | B2 (speaker2) | ✅ | 33.3% | Yo no sé. | yo no |
| u068 | 104.4-110.3 | ✔ | A2 | B2 (speaker2) | ❌ wrong | 90.0% | Ah, ok. Cuando vaya a Colombia, yo te enseño, ¿ok? | tengo okay hmm con ella colombia you're just |
| u069 | 110.7-111.4 |  | B2 | B2 (speaker2) | ✅ | 100.0% | Ok. | saying |
| u070 | 112.3-112.7 |  | A2 | B2 (speaker2) | ❌ wrong | 100.0% | ¡Chau! | okay |
| u071 | 113.6-115.7 |  | B2 | B2 (speaker2) | ✅ | 100.0% | ¿Ya? ¿Cómo? | okay yeah |
| u072 | 113.8-114.5 |  | A2 | unknown (unknown) | ❔ unknown | 100.0% | Chao. | oh |
| u073 | 116.8-117.6 |  | A | A (speaker4) | ✅ | 0.0% | Bueno, mi vida. | bueno mi vida |
| u074 | 118.0-118.5 |  | B2 | A (speaker4) | ❌ wrong | 0.0% | Bueno. | bueno |
| u075 | 118.5-120.0 | ✔ | A | A (speaker4) | ✅ | 0.0% | Entonces me le mando un beso a... | entonces me le mando un beso a |

## callhome_spa_snr20__spa_0018_2spk_snr20

Speaker mapping: `{"speaker1": "A", "speaker2": "B"}`

| id | time | long | GT spk | pred spk | speaker | WER | GT text | ASR text |
|---|---|---|---|---|---|---|---|---|
| u000 | 0.0-1.7 | ✔ | B | unknown (unknown) | ❔ unknown | 11.1% | ¿Se vais a ir el 31 vosotros? | vais a ir el treinta y un vosotros |
| u001 | 2.4-5.5 |  | A | A (speaker1) | ✅ | 0.0% | Mira, ya veremos, ya veremos. | mira ya veremos ya veremos |
| u002 | 5.0-10.2 | ✔ | B | B (speaker2) | ✅ | 28.6% | Oye, hoy Pili ha estado diciendo aquí que el tren de... Pili Álava está aquí, ha venido la boda de Miquel. | hoy pili ha estado diciendo aqui que el prendepa pilialaba esta aqui ha venido la boda de mickey |
| u003 | 10.3-11.6 |  | A | A (speaker1) | ✅ | 33.3% | ¡Ah, sí! ¡Miquel! | ah si |
| u004 | 11.5-13.2 |  | B | B (speaker2) | ✅ | 16.7% | Miquel se ha casado con Nicole. | mikel se ha casado con nicole |
| u005 | 13.6-14.3 |  | A | B (speaker2) | ❌ wrong | 0.0% | ¡Así! | asi |
| u006 | 14.2-15.1 |  | B |  | ∅ missed | 100.0% | Colchonchon |  |
| u007 | 15.4-16.9 | ✔ | A | B (speaker2) | ❌ wrong | 28.6% | Sí, sí, yo sé quien es Nicole. | si si yo se que eres nicole |
| u008 | 16.9-20.5 | ✔ | B | B (speaker2) | ✅ | 91.7% | Sí, se ha casado por lo civil porque ella es judía, ¿no? | porque |
| u009 | 20.6-21.6 |  | A | B (speaker2) | ❌ wrong | 0.0% | Ah, no sabía. | ah no sabia |
| u010 | 21.6-26.2 | ✔ | B | B (speaker2) | ✅ | 14.3% | Sí, entonces se han casado por lo civil, pero una boda estrictamente familiar, ¿no? | si entonces se han casado por los civiles pero una boda estrictamente familiar no |
| u011 | 26.0-27.4 |  | A | A (speaker1) | ✅ | 0.0% | Imagínate | imaginate |
| u012 | 26.9-37.4 | ✔ | B | B (speaker2) | ✅ | 9.4% | Sí, y se casó el martes. Y entonces ha venido Begochu, su marido, y Pili. Y hoy se han ido Begochu y su marido y se ha quedado Pili hasta el domingo. | si y se caso el martes y entonces ha venido begochu el marido y pili y hoy hoy se han ido begoche y su marido y se ha quedado pili hasta el domingo |
| u013 | 37.5-38.2 |  | A |  | ∅ missed | 100.0% | Está bien. |  |
| u014 | 37.9-44.9 | ✔ | B | B (speaker2) | ✅ | 75.0% | Entonces nos estaba diciendo que hay un tren de París a Vendaya. | de paris a |
| u015 | 43.5-44.3 |  | A | A (speaker1) | ✅ | 100.0% | En Daya. | a hendalla |
| u016 | 45.2-46.7 |  | B | B (speaker2) | ✅ | 0.0% | Que es una maravilla, ¿vale? | que es una maravilla vale |
| u017 | 46.6-48.5 |  | A | A (speaker1) | ✅ | 0.0% | No, sí, nosotros... | no si nosotros |
| u018 | 47.1-49.9 |  | B | A (speaker1) | ❌ wrong | 50.0% | 3 o 4 horas | cuatro horas |
| u019 | 49.9-52.7 | ✔ | A | A (speaker1) | ✅ | 12.5% | Sí, bueno, no, no, no tanto, no tanto. | si bueno no no no tanto no tanto no |
| u020 | 52.2-53.6 |  | B | A (speaker1) | ❌ wrong | 0.0% | Cuatro horas, ha dicho. | cuatro horas ha dicho |
| u021 | 53.9-59.2 | ✔ | A | A (speaker1) | ✅ | 5.9% | Bueno, ya es un poco más. Ese es el mismo tren que nosotros agarramos cuando fuimos ama. | bueno ya es un poco mas ese es el mismo tren que nosotros agarramos cuando fuimos |
| u022 | 58.2-60.4 |  | B | A (speaker1) | ❌ wrong | 83.3% | Si, ¿Cuál es el Tafo eso? | el |
| u023 | 60.3-61.3 |  | A | A (speaker1) | ✅ | 50.0% | El TDB | tdv tdb |
| u024 | 61.9-62.7 |  | B | A (speaker1) | ❌ wrong | 100.0% | ¿TDB? | tdv |
| u025 | 62.8-65.7 |  | A | A (speaker1) | ✅ | 66.7% | TDB, TGV, TREN. | t tgb tren |
| u026 | 65.2-65.9 |  | B |  | ∅ missed | 100.0% | TGV |  |
| u027 | 65.7-67.5 |  | A | A (speaker1) | ✅ | 0.0% | Tren de gran velocidad. | tren de gran velocidad |
| u028 | 67.2-71.4 | ✔ | B |  | ∅ missed | 100.0% | ¡Ah, el tac sí, sí, sí! ¡Tec, tec, ya, ya! |  |
| u029 | 69.0-72.6 |  | A |  | ∅ missed | 100.0% | Tec. |  |
| u030 | 72.1-73.5 |  | B | A (speaker1) | ❌ wrong | 100.0% | Tec Tec | t |
| u031 | 73.4-75.6 |  | A | A (speaker1) | ✅ | 0.0% | T-G-B | t g b |
| u032 | 75.6-78.0 |  | B | A (speaker1) | ❌ wrong | 33.3% | TGB, ajá, ya. | aja ya |
| u033 | 76.6-79.8 |  | A | A (speaker1) | ✅ | 20.0% | El tren de gran vitez | tren de gran vitez |
| u034 | 79.7-81.6 |  | B | A (speaker1) | ❌ wrong | 100.0% | ¡Gracias! | aja |
| u035 | 81.2-82.8 |  | A | A (speaker1) | ✅ | 0.0% | Eso es. | eso es |
| u036 | 82.1-83.4 |  | B | A (speaker1) | ❌ wrong | 25.0% | Ya, ya, ya. ¿Y? | ya ya ya |
| u037 | 83.7-86.0 | ✔ | A | A (speaker1) | ✅ | 0.0% | No, pero ese dura como cinco horas. | no pero ese dura como cinco horas |
| u038 | 86.3-86.9 |  | B | A (speaker1) | ❌ wrong | 100.0% | a sí?. | eh |
| u039 | 87.1-88.5 |  | A | A (speaker1) | ✅ | 0.0% | Sí, pero bueno, claro. | si pero bueno claro |
| u040 | 87.9-90.9 |  | B |  | ∅ missed | 100.0% | Pero vale la pena. |  |
| u041 | 89.2-92.7 | ✔ | A |  | ∅ missed | 100.0% | No, por supuesto que vale la pena, por supuesto que vale la pena. |  |
| u042 | 92.9-93.4 |  | B |  | ∅ missed | 100.0% | pero |  |
| u043 | 93.2-107.3 | ✔ | A | A (speaker1) | ✅ | 4.5% | Pero claro, hay que ver, ¿no? Hay que ver, o sea, yo por supuesto que si yo veo que los días que vamos a estar en París, porque ahora el problema que se me pudiera presentar es que del 24 al 31, | pero claro hay que ver no hay que ver o sea yo yo por supuesto que si si yo veo que los dias que vamos a estar en paris porque ahora el problema que se me pudiera presentar es que del veinticuatro al treinta y un |
| u044 | 107.4-108.0 |  | B |  | ∅ missed | 100.0% | hmm |  |
| u045 | 107.7-111.6 | ✔ | A | A (speaker1) | ✅ | 0.0% | En París no sé cómo estará el movimiento de trabajo. | en paris no se como estara el movimiento de trabajo |
| u046 | 110.8-112.3 |  | B |  | ∅ missed | 100.0% | De apartamentos ya. |  |
| u047 | 112.4-115.6 | ✔ | A | A (speaker1) | ✅ | 12.5% | Yo supongo que un inmobiliario debe trabajar, ¿no? | yo supongo que un inmobiliario debe trabajar si |
| u048 | 115.1-117.0 |  | B | A (speaker1) | ❌ wrong | 0.0% | Sí, sí, sí. | si si si |
| u049 | 116.5-117.6 |  | A | A (speaker1) | ✅ | 0.0% | Así que... | asi que |
| u050 | 117.5-120.0 | ✔ | B | B (speaker2) | ✅ | 9.1% | Le voy a decir una cosa, París, ¿qué es el lugar? | te voy a decir una cosa paris que es el lugar |
