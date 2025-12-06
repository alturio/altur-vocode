import re
from datetime import datetime
from typing import List, Optional, Tuple

import dateparser
from dateparser.search import search_dates
from loguru import logger

# Minimum length for a date match to be considered valid
# This filters out false positives like "mi", "ma", etc.
MIN_DATE_MATCH_LENGTH = 4

# Known date-related words/patterns in Spanish, English, and Portuguese that are valid short matches
VALID_SHORT_PATTERNS = {
    "es": {
        # Days
        "hoy", "ayer", "anteayer", "mañana", "pasado mañana",
        # Weekdays
        "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
        # Relative time
        "semana", "mes", "año", "hora", "minuto", "momento",
        "ahora", "ahorita", "luego", "después", "antes",
    },
    "en": {
        # Days
        "today", "tomorrow", "yesterday",
        # Weekdays
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        # Relative time
        "week", "month", "year", "hour", "minute", "moment",
        "now", "later", "soon", "before", "after",
    },
    "pt": {
        # Days
        "hoje", "ontem", "anteontem", "amanhã", "depois de amanhã",
        # Weekdays
        "segunda", "segunda-feira", "terça", "terça-feira", 
        "quarta", "quarta-feira", "quinta", "quinta-feira",
        "sexta", "sexta-feira", "sábado", "domingo",
        # Relative time
        "semana", "mês", "ano", "hora", "minuto", "momento",
        "agora", "logo", "depois", "antes", "já",
    },
}

# Words that indicate past tense - check AFTER the date word
PAST_MODIFIERS_AFTER = {
    "es": {
        "pasado", "pasada", "pasados", "pasadas",
        "anterior", "anteriores",
        "antepasado", "antepasada",
        "atrás",  # "dos semanas atrás"
    },
    "en": {
        "ago",  # "two weeks ago"
        "back",  # "a week back"
        "earlier",  # "a month earlier"
        "before",  # "the day before"
        "prior",  # "the week prior"
    },
    "pt": {
        "passado", "passada", "passados", "passadas",
        "anterior", "anteriores",
        "atrás",  # "duas semanas atrás"
        "retrasado", "retrasada",  # "semana retrasada"
    },
}

# Words that indicate past tense - check BEFORE the date word
PAST_MODIFIERS_BEFORE = {
    "es": {
        # Direct modifiers
        "el pasado", "la pasada", "los pasados", "las pasadas",
        "el último", "la última", "los últimos", "las últimas",
        "el anterior", "la anterior", "los anteriores", "las anteriores",
        "el otro", "la otra",  # "el otro día"
        # "Hace" constructions (ago)
        "hace",  # "hace una semana", "hace dos días"
        # "Ya" indicating completed action
        "ya",  # "ya el lunes"
    },
    "en": {
        "last", "past", "previous", "prior",
        "the other",  # "the other day"
        "the previous", "the past",
        # Ago constructions (word order: "a week ago")
        "a",  # part of "a week ago", "a month ago"
    },
    "pt": {
        # Direct modifiers
        "o passado", "a passada", "os passados", "as passadas",
        "o último", "a última", "os últimos", "as últimas",
        "o anterior", "a anterior", "os anteriores", "as anteriores",
        "o outro", "a outra",  # "o outro dia"
        "no passado",  # "no passado"
        # "Há" / "Faz" constructions (ago)
        "há",  # "há uma semana", "há dois dias"
        "faz",  # "faz uma semana"
        # "Já" indicating completed action
        "já",  # "já na segunda"
    },
}

# Words that indicate future tense - check AFTER the date word
FUTURE_MODIFIERS_AFTER = {
    "es": {
        "que viene", "que entra", "que sigue",
        "próximo", "próxima", "próximos", "próximas",
        "siguiente", "siguientes",
        "entrante", "entrantes",
    },
    "en": {
        "next",  # sometimes comes after: "Monday next"
        "coming", "upcoming", "following",
        "from now",  # "two weeks from now"
        "later",  # "a week later"
        "ahead",  # "the week ahead"
    },
    "pt": {
        "que vem", "que entra",
        "próximo", "próxima", "próximos", "próximas",
        "seguinte", "seguintes",
    },
}

# Words that indicate future tense - check BEFORE the date word
FUTURE_MODIFIERS_BEFORE = {
    "es": {
        # Direct modifiers
        "el próximo", "la próxima", "los próximos", "las próximas",
        "el siguiente", "la siguiente", "los siguientes", "las siguientes",
        "el entrante", "la entrante",
        "este", "esta", "estos", "estas",  # "este lunes", "esta semana"
        # "Dentro de" (in/within)
        "dentro de",  # "dentro de una semana"
        "en",  # "en una semana", "en dos días"
        # "Para" (by/for)
        "para",  # "para el lunes"
        # "El/la que viene"
        "el que viene", "la que viene",
    },
    "en": {
        "next", "this", "the next", "the coming", "the upcoming", "the following",
        # "In" constructions
        "in",  # "in a week", "in two days"
        "in a", "in an", "in the",
        # "Within" constructions
        "within", "within a", "within the",
        # "By" constructions
        "by", "by the", "by next",
        # "For" constructions (deadline)
        "for", "for the", "for next",
    },
    "pt": {
        # Direct modifiers
        "o próximo", "a próxima", "os próximos", "as próximas",
        "o seguinte", "a seguinte", "os seguintes", "as seguintes",
        "este", "esta", "estes", "estas",  # "esta segunda", "esta semana"
        "esse", "essa", "esses", "essas",
        "neste", "nesta", "nestes", "nestas",  # "nesta semana"
        # "Dentro de" / "Daqui a" (in/within)
        "dentro de",  # "dentro de uma semana"
        "daqui a",  # "daqui a dois dias"
        "em",  # "em uma semana"
        # "Para" / "Até" (by/for)
        "para",  # "para segunda"
        "até",  # "até sexta"
        # "O/a que vem"
        "o que vem", "a que vem",
        "na próxima", "no próximo",
    },
}

# Common past tense verbs/patterns to detect in sentence context
PAST_TENSE_VERBS = {
    "es": {
        # Preterite (pretérito indefinido) - irregular verbs
        "fui", "fue", "fueron", "fuimos", "fuiste",
        "estuve", "estuvo", "estuvieron", "estuvimos", "estuviste",
        "hice", "hizo", "hicieron", "hicimos", "hiciste",
        "tuve", "tuvo", "tuvieron", "tuvimos", "tuviste",
        "pude", "pudo", "pudieron", "pudimos", "pudiste",
        "dije", "dijo", "dijeron", "dijimos", "dijiste",
        "vine", "vino", "vinieron", "vinimos", "viniste",
        "supe", "supo", "supieron", "supimos", "supiste",
        "puse", "puso", "pusieron", "pusimos", "pusiste",
        "quise", "quiso", "quisieron", "quisimos", "quisiste",
        "traje", "trajo", "trajeron", "trajimos", "trajiste",
        "conduje", "condujo", "condujeron", "condujimos",
        "anduve", "anduvo", "anduvieron", "anduvimos",
        "cupe", "cupo", "cupieron", "cupimos",
        "di", "dio", "dieron", "dimos", "diste",
        "vi", "vio", "vieron", "vimos", "viste",
        # Common regular preterite - AR verbs
        "pagué", "pagó", "pagaron", "pagamos", "pagaste",
        "llamé", "llamó", "llamaron", "llamamos", "llamaste",
        "hablé", "habló", "hablaron", "hablamos", "hablaste",
        "compré", "compró", "compraron", "compramos", "compraste",
        "llegué", "llegó", "llegaron", "llegamos", "llegaste",
        "pasé", "pasó", "pasaron", "pasamos", "pasaste",
        "trabajé", "trabajó", "trabajaron", "trabajamos",
        "jugué", "jugó", "jugaron", "jugamos",
        "busqué", "buscó", "buscaron", "buscamos",
        "empecé", "empezó", "empezaron", "empezamos",
        "almorcé", "almorzó", "almorzaron", "almorzamos",
        "comencé", "comenzó", "comenzaron", "comenzamos",
        "saqué", "sacó", "sacaron", "sacamos",
        "toqué", "tocó", "tocaron", "tocamos",
        "practiqué", "practicó", "practicaron",
        "expliqué", "explicó", "explicaron",
        "marqué", "marcó", "marcaron",
        "entregué", "entregó", "entregaron",
        "navegué", "navegó", "navegaron",
        "colgué", "colgó", "colgaron",
        "apagué", "apagó", "apagaron",
        "pagué", "pagó", "pagaron",
        "cerré", "cerró", "cerraron",
        "esperé", "esperó", "esperaron",
        "terminé", "terminó", "terminaron",
        "envié", "envió", "enviaron",
        "mandé", "mandó", "mandaron",
        "dejé", "dejó", "dejaron",
        "entré", "entró", "entraron",
        "salí", "salió", "salieron",
        "abrí", "abrió", "abrieron",
        "recibí", "recibió", "recibieron",
        "viví", "vivió", "vivieron",
        "escribí", "escribió", "escribieron",
        "subí", "subió", "subieron",
        "decidí", "decidió", "decidieron",
        "pedí", "pidió", "pidieron",
        "dormí", "durmió", "durmieron",
        "sentí", "sintió", "sintieron",
        "seguí", "siguió", "siguieron",
        "repetí", "repitió", "repitieron",
        "preferí", "prefirió", "prefirieron",
        "mentí", "mintió", "mintieron",
        "morí", "murió", "murieron",
        # Common -ER/-IR preterite
        "comí", "comió", "comieron", "comimos",
        "bebí", "bebió", "bebieron", "bebimos",
        "corrí", "corrió", "corrieron", "corrimos",
        "leí", "leyó", "leyeron", "leímos",
        "creí", "creyó", "creyeron", "creímos",
        "oí", "oyó", "oyeron", "oímos",
        "caí", "cayó", "cayeron", "caímos",
        "construí", "construyó", "construyeron",
        "destruí", "destruyó", "destruyeron",
        "huí", "huyó", "huyeron",
        # Imperfect tense
        "era", "eras", "éramos", "eran",
        "iba", "ibas", "íbamos", "iban",
        "estaba", "estabas", "estábamos", "estaban",
        "había", "habías", "habíamos", "habían",
        "tenía", "tenías", "teníamos", "tenían",
        "podía", "podías", "podíamos", "podían",
        "quería", "querías", "queríamos", "querían",
        "sabía", "sabías", "sabíamos", "sabían",
        "decía", "decías", "decíamos", "decían",
        "hacía", "hacías", "hacíamos", "hacían",
        "veía", "veías", "veíamos", "veían",
        "venía", "venías", "veníamos", "venían",
        "vivía", "vivías", "vivíamos", "vivían",
        "trabajaba", "trabajabas", "trabajábamos", "trabajaban",
        "hablaba", "hablabas", "hablábamos", "hablaban",
        "llamaba", "llamabas", "llamábamos", "llamaban",
        "pagaba", "pagabas", "pagábamos", "pagaban",
        "jugaba", "jugabas", "jugábamos", "jugaban",
        "comía", "comías", "comíamos", "comían",
        "salía", "salías", "salíamos", "salían",
        "dormía", "dormías", "dormíamos", "dormían",
        # Compound past (pretérito perfecto)
        "he ido", "has ido", "ha ido", "hemos ido", "han ido",
        "he estado", "has estado", "ha estado", "hemos estado", "han estado",
        "he hecho", "has hecho", "ha hecho", "hemos hecho", "han hecho",
        "he tenido", "has tenido", "ha tenido", "hemos tenido", "han tenido",
        "he visto", "has visto", "ha visto", "hemos visto", "han visto",
        "he dicho", "has dicho", "ha dicho", "hemos dicho", "han dicho",
        "he pagado", "has pagado", "ha pagado", "hemos pagado", "han pagado",
        "he llamado", "has llamado", "ha llamado", "hemos llamado", "han llamado",
        "he comprado", "has comprado", "ha comprado", "hemos comprado", "han comprado",
        "he trabajado", "has trabajado", "ha trabajado", "hemos trabajado", "han trabajado",
        "he comido", "has comido", "ha comido", "hemos comido", "han comido",
        "he salido", "has salido", "ha salido", "hemos salido", "han salido",
        "he venido", "has venido", "ha venido", "hemos venido", "han venido",
        # Pluperfect (pretérito pluscuamperfecto)
        "había ido", "había estado", "había hecho", "había tenido",
        "había visto", "había dicho", "había pagado", "había llamado",
    },
    "en": {
        # Common irregular past tense
        "went", "was", "were", "did", "had", "got", "came", "made",
        "said", "told", "gave", "took", "saw", "knew", "thought",
        "found", "left", "felt", "became", "brought", "began",
        "kept", "held", "wrote", "stood", "heard", "let", "meant",
        "set", "met", "ran", "paid", "sat", "spoke", "lay", "led",
        "read", "grew", "lost", "fell", "sent", "built", "spent",
        "cut", "hit", "put", "shut", "hurt", "cost", "quit",
        "chose", "broke", "drove", "ate", "flew", "forgot", "froze",
        "hid", "rode", "rang", "sang", "sank", "wore", "woke", "won",
        "drew", "drank", "swam", "threw", "blew", "knew", "grew",
        "slept", "swept", "wept", "crept", "kept", "leapt",
        "dealt", "dreamt", "learnt", "burnt", "spelt",
        "bought", "caught", "taught", "fought", "sought", "brought",
        # Common regular past tense (-ed)
        "called", "talked", "walked", "worked", "played", "wanted",
        "needed", "started", "finished", "happened", "arrived",
        "passed", "asked", "answered", "helped", "stopped", "tried",
        "used", "moved", "lived", "loved", "liked", "looked",
        "waited", "watched", "listened", "opened", "closed", "turned",
        "changed", "followed", "showed", "reached", "returned",
        "remembered", "believed", "received", "decided", "expected",
        "visited", "traveled", "travelled", "ordered", "delivered",
        "scheduled", "attended", "completed", "submitted", "reviewed",
        # Past continuous
        "was going", "were going", "was doing", "were doing",
        "was working", "were working", "was waiting", "were waiting",
        # Present perfect (indicates completed action)
        "have been", "has been", "have gone", "has gone",
        "have done", "has done", "have made", "has made",
        "have had", "has had", "have seen", "has seen",
        "have said", "has said", "have paid", "has paid",
        "have called", "has called", "have finished", "has finished",
        # Past perfect
        "had been", "had gone", "had done", "had made",
        "had had", "had seen", "had said", "had paid",
        # Common past expressions
        "already", "just", "recently", "earlier",
    },
    "pt": {
        # Pretérito perfeito (simple past) - irregular verbs
        "fui", "foi", "foram", "fomos", "foste",
        "estive", "esteve", "estiveram", "estivemos", "estiveste",
        "fiz", "fez", "fizeram", "fizemos", "fizeste",
        "tive", "teve", "tiveram", "tivemos", "tiveste",
        "pude", "pôde", "puderam", "pudemos", "pudeste",
        "disse", "disseram", "dissemos", "disseste",
        "vim", "veio", "vieram", "viemos", "vieste",
        "soube", "souberam", "soubemos", "soubeste",
        "pus", "pôs", "puseram", "pusemos", "puseste",
        "quis", "quiseram", "quisemos", "quiseste",
        "trouxe", "trouxeram", "trouxemos", "trouxeste",
        "dei", "deu", "deram", "demos", "deste",
        "vi", "viu", "viram", "vimos", "viste",
        "li", "leu", "leram", "lemos", "leste",
        "ouvi", "ouviu", "ouviram", "ouvimos", "ouviste",
        # Common regular preterite - AR verbs
        "paguei", "pagou", "pagaram", "pagamos", "pagaste",
        "liguei", "ligou", "ligaram", "ligamos", "ligaste",
        "chamei", "chamou", "chamaram", "chamamos", "chamaste",
        "falei", "falou", "falaram", "falamos", "falaste",
        "comprei", "comprou", "compraram", "compramos", "compraste",
        "cheguei", "chegou", "chegaram", "chegamos", "chegaste",
        "passei", "passou", "passaram", "passamos", "passaste",
        "trabalhei", "trabalhou", "trabalharam", "trabalhamos",
        "joguei", "jogou", "jogaram", "jogamos",
        "busquei", "buscou", "buscaram", "buscamos",
        "comecei", "começou", "começaram", "começamos",
        "almocei", "almoçou", "almoçaram", "almoçamos",
        "terminei", "terminou", "terminaram", "terminamos",
        "enviei", "enviou", "enviaram", "enviamos",
        "mandei", "mandou", "mandaram", "mandamos",
        "deixei", "deixou", "deixaram", "deixamos",
        "entrei", "entrou", "entraram", "entramos",
        "voltei", "voltou", "voltaram", "voltamos",
        "encontrei", "encontrou", "encontraram", "encontramos",
        "acordei", "acordou", "acordaram", "acordamos",
        "fechei", "fechou", "fecharam", "fechamos",
        "esperei", "esperou", "esperaram", "esperamos",
        # Common -ER/-IR preterite
        "comi", "comeu", "comeram", "comemos",
        "bebi", "bebeu", "beberam", "bebemos",
        "corri", "correu", "correram", "corremos",
        "escrevi", "escreveu", "escreveram", "escrevemos",
        "vivi", "viveu", "viveram", "vivemos",
        "dormi", "dormiu", "dormiram", "dormimos",
        "parti", "partiu", "partiram", "partimos",
        "abri", "abriu", "abriram", "abrimos",
        "decidi", "decidiu", "decidiram", "decidimos",
        "recebi", "recebeu", "receberam", "recebemos",
        "senti", "sentiu", "sentiram", "sentimos",
        "pedi", "pediu", "pediram", "pedimos",
        "segui", "seguiu", "seguiram", "seguimos",
        "saí", "saiu", "saíram", "saímos",
        # Imperfect tense (pretérito imperfeito)
        "era", "eras", "éramos", "eram",
        "ia", "ias", "íamos", "iam",
        "estava", "estavas", "estávamos", "estavam",
        "tinha", "tinhas", "tínhamos", "tinham",
        "havia", "havias", "havíamos", "haviam",
        "podia", "podias", "podíamos", "podiam",
        "queria", "querias", "queríamos", "queriam",
        "sabia", "sabias", "sabíamos", "sabiam",
        "dizia", "dizias", "dizíamos", "diziam",
        "fazia", "fazias", "fazíamos", "faziam",
        "via", "vias", "víamos", "viam",
        "vinha", "vinhas", "vínhamos", "vinham",
        "vivia", "vivias", "vivíamos", "viviam",
        "trabalhava", "trabalhavas", "trabalhávamos", "trabalhavam",
        "falava", "falavas", "falávamos", "falavam",
        "chamava", "chamavas", "chamávamos", "chamavam",
        "pagava", "pagavas", "pagávamos", "pagavam",
        "jogava", "jogavas", "jogávamos", "jogavam",
        "comia", "comias", "comíamos", "comiam",
        "saía", "saías", "saíamos", "saíam",
        "dormia", "dormias", "dormíamos", "dormiam",
        # Compound past (pretérito perfeito composto)
        "tenho ido", "tens ido", "tem ido", "temos ido", "têm ido",
        "tenho estado", "tens estado", "tem estado", "temos estado", "têm estado",
        "tenho feito", "tens feito", "tem feito", "temos feito", "têm feito",
        "tenho tido", "tens tido", "tem tido", "temos tido", "têm tido",
        "tenho visto", "tens visto", "tem visto", "temos visto", "têm visto",
        "tenho dito", "tens dito", "tem dito", "temos dito", "têm dito",
        "tenho pago", "tens pago", "tem pago", "temos pago", "têm pago",
        "tenho ligado", "tens ligado", "tem ligado", "temos ligado", "têm ligado",
        "tenho comprado", "tens comprado", "tem comprado", "temos comprado",
        "tenho trabalhado", "tens trabalhado", "tem trabalhado", "temos trabalhado",
        "tenho comido", "tens comido", "tem comido", "temos comido",
        "tenho saído", "tens saído", "tem saído", "temos saído",
        "tenho vindo", "tens vindo", "tem vindo", "temos vindo",
        # Pluperfect (mais-que-perfeito)
        "tinha ido", "tinha estado", "tinha feito", "tinha tido",
        "tinha visto", "tinha dito", "tinha pago", "tinha ligado",
        "havia ido", "havia estado", "havia feito", "havia tido",
    },
}

# Common future/present tense verbs/patterns to detect in sentence context
FUTURE_TENSE_VERBS = {
    "es": {
        # Present tense (often used for near future in Spanish)
        "voy", "vas", "va", "vamos", "van",
        "tengo", "tienes", "tiene", "tenemos", "tienen",
        "puedo", "puedes", "puede", "podemos", "pueden",
        "quiero", "quieres", "quiere", "queremos", "quieren",
        "necesito", "necesitas", "necesita", "necesitamos", "necesitan",
        "debo", "debes", "debe", "debemos", "deben",
        "espero", "esperas", "espera", "esperamos", "esperan",
        "pienso", "piensas", "piensa", "pensamos", "piensan",
        "planeo", "planeas", "planea", "planeamos", "planean",
        "pago", "pagas", "paga", "pagamos", "pagan",
        "llamo", "llamas", "llama", "llamamos", "llaman",
        "hago", "haces", "hace", "hacemos", "hacen",
        "vengo", "vienes", "viene", "venimos", "vienen",
        "salgo", "sales", "sale", "salimos", "salen",
        "llego", "llegas", "llega", "llegamos", "llegan",
        "empiezo", "empiezas", "empieza", "empezamos", "empiezan",
        "termino", "terminas", "termina", "terminamos", "terminan",
        "trabajo", "trabajas", "trabaja", "trabajamos", "trabajan",
        "estudio", "estudias", "estudia", "estudiamos", "estudian",
        "compro", "compras", "compra", "compramos", "compran",
        "viajo", "viajas", "viaja", "viajamos", "viajan",
        "regreso", "regresas", "regresa", "regresamos", "regresan",
        "visito", "visitas", "visita", "visitamos", "visitan",
        "asisto", "asistes", "asiste", "asistimos", "asisten",
        "me reúno", "te reúnes", "se reúne", "nos reunimos", "se reúnen",
        # Future tense (futuro simple)
        "iré", "irás", "irá", "iremos", "irán",
        "seré", "serás", "será", "seremos", "serán",
        "estaré", "estarás", "estará", "estaremos", "estarán",
        "tendré", "tendrás", "tendrá", "tendremos", "tendrán",
        "haré", "harás", "hará", "haremos", "harán",
        "podré", "podrás", "podrá", "podremos", "podrán",
        "sabré", "sabrás", "sabrá", "sabremos", "sabrán",
        "vendré", "vendrás", "vendrá", "vendremos", "vendrán",
        "saldré", "saldrás", "saldrá", "saldremos", "saldrán",
        "diré", "dirás", "dirá", "diremos", "dirán",
        "querré", "querrás", "querrá", "querremos", "querrán",
        "pagaré", "pagarás", "pagará", "pagaremos", "pagarán",
        "llamaré", "llamarás", "llamará", "llamaremos", "llamarán",
        "trabajaré", "trabajarás", "trabajará", "trabajaremos", "trabajarán",
        "llegaré", "llegarás", "llegará", "llegaremos", "llegarán",
        "empezaré", "empezarás", "empezará", "empezaremos", "empezarán",
        "terminaré", "terminarás", "terminará", "terminaremos", "terminarán",
        "compraré", "comprarás", "comprará", "compraremos", "comprarán",
        "viajaré", "viajarás", "viajará", "viajaremos", "viajarán",
        "regresaré", "regresarás", "regresará", "regresaremos", "regresarán",
        "visitaré", "visitarás", "visitará", "visitaremos", "visitarán",
        "asistiré", "asistirás", "asistirá", "asistiremos", "asistirán",
        "comeré", "comerás", "comerá", "comeremos", "comerán",
        "viviré", "vivirás", "vivirá", "viviremos", "vivirán",
        "escribiré", "escribirás", "escribirá", "escribiremos", "escribirán",
        # Periphrastic future (ir a + infinitive)
        "voy a", "vas a", "va a", "vamos a", "van a",
        # Conditional (often used for polite future)
        "iría", "sería", "estaría", "tendría", "haría",
        "podría", "querría", "debería", "pagaría", "llamaría",
        # Common future expressions
        "me gustaría", "quisiera", "necesitaría",
    },
    "en": {
        # Simple future with will
        "will", "will be", "will go", "will do", "will make",
        "will have", "will see", "will get", "will come", "will take",
        "will give", "will find", "will know", "will think", "will want",
        "will need", "will pay", "will call", "will work", "will start",
        "will finish", "will arrive", "will leave", "will return",
        "will meet", "will visit", "will travel", "will attend",
        # Going to future
        "going to", "am going to", "is going to", "are going to",
        "gonna", "am gonna", "is gonna", "are gonna",
        # Present continuous for future
        "am going", "is going", "are going",
        "am coming", "is coming", "are coming",
        "am leaving", "is leaving", "are leaving",
        "am meeting", "is meeting", "are meeting",
        "am visiting", "is visiting", "are visiting",
        "am working", "is working", "are working",
        "am traveling", "is traveling", "are traveling",
        "am starting", "is starting", "are starting",
        "am arriving", "is arriving", "are arriving",
        # Simple present for scheduled future
        "go", "goes", "leave", "leaves", "arrive", "arrives",
        "start", "starts", "begin", "begins", "end", "ends",
        "open", "opens", "close", "closes", "depart", "departs",
        # Modal verbs for future
        "shall", "should", "would", "could", "might", "may",
        "can", "must", "need to", "have to", "has to",
        # Common future expressions
        "plan to", "plans to", "planning to",
        "intend to", "intends to", "intending to",
        "expect to", "expects to", "expecting to",
        "hope to", "hopes to", "hoping to",
        "want to", "wants to", "wanting to",
        "need to", "needs to", "needing to",
        "about to", "ready to", "scheduled to",
        "tomorrow", "soon", "later", "next",
    },
    "pt": {
        # Present tense (often used for near future in Portuguese)
        "vou", "vais", "vai", "vamos", "vão",
        "tenho", "tens", "tem", "temos", "têm",
        "posso", "podes", "pode", "podemos", "podem",
        "quero", "queres", "quer", "queremos", "querem",
        "preciso", "precisas", "precisa", "precisamos", "precisam",
        "devo", "deves", "deve", "devemos", "devem",
        "espero", "esperas", "espera", "esperamos", "esperam",
        "penso", "pensas", "pensa", "pensamos", "pensam",
        "planejo", "planejas", "planeja", "planejamos", "planejam",
        "pago", "pagas", "paga", "pagamos", "pagam",
        "ligo", "ligas", "liga", "ligamos", "ligam",
        "chamo", "chamas", "chama", "chamamos", "chamam",
        "faço", "fazes", "faz", "fazemos", "fazem",
        "venho", "vens", "vem", "vimos", "vêm",
        "saio", "sais", "sai", "saímos", "saem",
        "chego", "chegas", "chega", "chegamos", "chegam",
        "começo", "começas", "começa", "começamos", "começam",
        "termino", "terminas", "termina", "terminamos", "terminam",
        "trabalho", "trabalhas", "trabalha", "trabalhamos", "trabalham",
        "estudo", "estudas", "estuda", "estudamos", "estudam",
        "compro", "compras", "compra", "compramos", "compram",
        "viajo", "viajas", "viaja", "viajamos", "viajam",
        "volto", "voltas", "volta", "voltamos", "voltam",
        "visito", "visitas", "visita", "visitamos", "visitam",
        "encontro", "encontras", "encontra", "encontramos", "encontram",
        # Future tense (futuro do presente)
        "irei", "irás", "irá", "iremos", "irão",
        "serei", "serás", "será", "seremos", "serão",
        "estarei", "estarás", "estará", "estaremos", "estarão",
        "terei", "terás", "terá", "teremos", "terão",
        "farei", "farás", "fará", "faremos", "farão",
        "poderei", "poderás", "poderá", "poderemos", "poderão",
        "saberei", "saberás", "saberá", "saberemos", "saberão",
        "virei", "virás", "virá", "viremos", "virão",
        "sairei", "sairás", "sairá", "sairemos", "sairão",
        "direi", "dirás", "dirá", "diremos", "dirão",
        "quererei", "quererás", "quererá", "quereremos", "quererão",
        "pagarei", "pagarás", "pagará", "pagaremos", "pagarão",
        "ligarei", "ligarás", "ligará", "ligaremos", "ligarão",
        "chamarei", "chamarás", "chamará", "chamaremos", "chamarão",
        "trabalharei", "trabalharás", "trabalhará", "trabalharemos", "trabalharão",
        "chegarei", "chegarás", "chegará", "chegaremos", "chegarão",
        "começarei", "começarás", "começará", "começaremos", "começarão",
        "terminarei", "terminarás", "terminará", "terminaremos", "terminarão",
        "comprarei", "comprarás", "comprará", "compraremos", "comprarão",
        "viajarei", "viajarás", "viajará", "viajaremos", "viajarão",
        "voltarei", "voltarás", "voltará", "voltaremos", "voltarão",
        "visitarei", "visitarás", "visitará", "visitaremos", "visitarão",
        "comerei", "comerás", "comerá", "comeremos", "comerão",
        "viverei", "viverás", "viverá", "viveremos", "viverão",
        "escreverei", "escreverás", "escreverá", "escreveremos", "escreverão",
        # Periphrastic future (ir + infinitive)
        "vou", "vais", "vai", "vamos", "vão",
        # Conditional (often used for polite future)
        "iria", "seria", "estaria", "teria", "faria",
        "poderia", "quereria", "deveria", "pagaria", "ligaria",
        # Common future expressions
        "gostaria", "precisaria", "queria",
        # Time expressions
        "amanhã", "logo", "depois", "em breve",
    },
}


def _is_valid_date_match(matched_text: str, text: str, languages: Optional[List[str]]) -> bool:
    """
    Check if a date match is valid (not a false positive).
    
    Filters out:
    - Matches that are too short and not known date words
    - Matches that are part of a larger word (not at word boundaries)
    """
    matched_lower = matched_text.lower().strip()
    
    # Check if it's a known valid short date word
    if languages:
        for lang in languages:
            if lang in VALID_SHORT_PATTERNS:
                if matched_lower in VALID_SHORT_PATTERNS[lang]:
                    return True
    
    # Reject matches that are too short
    if len(matched_lower) < MIN_DATE_MATCH_LENGTH:
        return False
    
    # Check word boundaries - the match should be at word boundaries in the original text
    escaped_match = re.escape(matched_text)
    pattern = rf'(?:^|(?<=[^\w])){escaped_match}(?=[^\w]|$)'
    if not re.search(pattern, text, re.IGNORECASE):
        if matched_text not in text:
            return False
    
    return True


def _get_temporal_direction(
    text: str, 
    match_pos: int, 
    match_text: str,
    languages: Optional[List[str]]
) -> Optional[str]:
    """
    Determine if the context around a date match indicates past or future.
    
    Checks:
    1. Direct modifiers before/after the date (e.g., "pasado", "próximo")
    2. Verb tense in the sentence (e.g., "fui" vs "voy")
    
    Returns:
        "past" if past indicators found
        "future" if future indicators found  
        None if no clear indication
    """
    text_lower = text.lower()
    match_end = match_pos + len(match_text)
    
    # Get immediate context before and after the match (up to 20 chars)
    context_before = text_lower[max(0, match_pos - 20):match_pos].strip()
    context_after = text_lower[match_end:match_end + 20].strip()
    
    langs = languages or ["es", "en"]
    
    # Priority 1: Check for direct modifiers (strongest signal)
    # Check for past modifiers
    for lang in langs:
        if lang in PAST_MODIFIERS_AFTER:
            for modifier in PAST_MODIFIERS_AFTER[lang]:
                if context_after.startswith(modifier):
                    return "past"
        if lang in PAST_MODIFIERS_BEFORE:
            for modifier in PAST_MODIFIERS_BEFORE[lang]:
                if context_before.endswith(modifier):
                    return "past"
    
    # Check for future modifiers
    for lang in langs:
        if lang in FUTURE_MODIFIERS_AFTER:
            for modifier in FUTURE_MODIFIERS_AFTER[lang]:
                if context_after.startswith(modifier):
                    return "future"
        if lang in FUTURE_MODIFIERS_BEFORE:
            for modifier in FUTURE_MODIFIERS_BEFORE[lang]:
                if context_before.endswith(modifier):
                    return "future"
    
    # Priority 2: Check for verb tense in the broader sentence
    # Get the full sentence/clause around the date (look for sentence boundaries)
    # Use a wider context window for verb detection
    sentence_start = max(0, match_pos - 50)
    sentence_end = min(len(text_lower), match_end + 50)
    sentence_context = text_lower[sentence_start:sentence_end]
    
    # Extract words from the sentence context
    words_in_context = set(re.findall(r'\b\w+\b', sentence_context))
    
    # Also check for multi-word patterns (like "voy a", "going to")
    past_verb_found = False
    future_verb_found = False
    
    for lang in langs:
        if lang in PAST_TENSE_VERBS:
            for verb in PAST_TENSE_VERBS[lang]:
                if " " in verb:
                    # Multi-word pattern
                    if verb in sentence_context:
                        past_verb_found = True
                        break
                else:
                    if verb in words_in_context:
                        past_verb_found = True
                        break
        
        if lang in FUTURE_TENSE_VERBS:
            for verb in FUTURE_TENSE_VERBS[lang]:
                if " " in verb:
                    # Multi-word pattern
                    if verb in sentence_context:
                        future_verb_found = True
                        break
                else:
                    if verb in words_in_context:
                        future_verb_found = True
                        break
    
    # If only one tense is found, use that
    if past_verb_found and not future_verb_found:
        return "past"
    if future_verb_found and not past_verb_found:
        return "future"
    
    # If both or neither found, return None (ambiguous)
    return None


def _parse_with_direction(
    match_text: str,
    direction: str,
    languages: Optional[List[str]],
    timezone: Optional[str],
) -> Optional[datetime]:
    """Re-parse a date match with the specified temporal direction."""
    settings = {
        "PREFER_DATES_FROM": direction,
    }
    if timezone:
        settings["TIMEZONE"] = timezone
    
    try:
        return dateparser.parse(match_text, languages=languages, settings=settings)
    except Exception:
        return None


def inject_parsed_dates(
    text: str,
    languages: Optional[List[str]] = None,
    timezone: Optional[str] = None,
) -> str:
    """
    Find date expressions in text and inject ISO formatted dates.

    Example:
        "voy a pagar el martes" -> "voy a pagar el martes (2025-12-09)"
        "el lunes pasado fui" -> "el lunes pasado (2025-12-01) fui"

    Args:
        text: The input text to parse for date expressions.
        languages: Optional list of language codes (e.g., ['es', 'en']).
        timezone: Optional timezone string (e.g., 'America/Mexico_City', 'UTC-6').

    Returns:
        The text with ISO formatted dates injected after detected date expressions.
    """
    base_settings = {}
    if timezone:
        base_settings["TIMEZONE"] = timezone

    try:
        # First pass: find all potential date matches without preference
        results = search_dates(
            text,
            languages=languages,
            settings=base_settings,
        )
    except Exception as e:
        logger.warning(f"Date parsing failed: {e}")
        return text

    if not results:
        return text

    # Filter out invalid matches and determine correct temporal direction
    valid_results: List[Tuple[str, datetime, int]] = []
    text_lower = text.lower()
    
    for matched_text, parsed_date in results:
        if not _is_valid_date_match(matched_text, text, languages):
            continue
        
        # Find position of this match
        pos = text_lower.find(matched_text.lower())
        if pos == -1:
            pos = text.find(matched_text)
        if pos == -1:
            continue
        
        # Check temporal context and re-parse if needed
        direction = _get_temporal_direction(text, pos, matched_text, languages)
        
        if direction:
            # Re-parse with the detected direction
            corrected_date = _parse_with_direction(
                matched_text, direction, languages, timezone
            )
            if corrected_date:
                parsed_date = corrected_date
        else:
            # Default to future for ambiguous cases (common in booking/payment contexts)
            corrected_date = _parse_with_direction(
                matched_text, "future", languages, timezone
            )
            if corrected_date:
                parsed_date = corrected_date
        
        valid_results.append((matched_text, parsed_date, pos))
    
    if not valid_results:
        return text

    # Sort by position in reverse order to avoid offset issues when replacing
    valid_results.sort(key=lambda x: x[2], reverse=True)
    
    enriched_text = text
    for matched_text, parsed_date, pos in valid_results:
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        
        # Find where to inject - after the match AND any trailing modifier
        match_end = pos + len(matched_text)
        
        # Check if there's a modifier after the match that should be included
        text_after = text[match_end:match_end + 15].lower().strip()
        modifier_length = 0
        
        langs = languages or ["es", "en"]
        for lang in langs:
            if lang in PAST_MODIFIERS_AFTER:
                for modifier in PAST_MODIFIERS_AFTER[lang]:
                    if text_after.startswith(modifier):
                        # Include the modifier and any space before it
                        space_match = re.match(r'\s*', text[match_end:])
                        space_len = len(space_match.group()) if space_match else 0
                        modifier_length = max(modifier_length, space_len + len(modifier))
            if lang in FUTURE_MODIFIERS_AFTER:
                for modifier in FUTURE_MODIFIERS_AFTER[lang]:
                    if text_after.startswith(modifier):
                        space_match = re.match(r'\s*', text[match_end:])
                        space_len = len(space_match.group()) if space_match else 0
                        modifier_length = max(modifier_length, space_len + len(modifier))
        
        # Build the injection
        full_match_end = match_end + modifier_length
        original_expression = text[pos:full_match_end]
        injection = f"{original_expression} ({formatted_date})"
        
        enriched_text = (
            enriched_text[:pos]
            + injection
            + enriched_text[full_match_end:]
        )

    logger.debug(f"Date parsing: '{text}' -> '{enriched_text}'")
    return enriched_text
