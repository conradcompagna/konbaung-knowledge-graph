You are performing an additive enrichment pass over existing sentence-level triples from the Konbaung Chronicle.

The text inside `ORIGINAL_V1_ANNOTATION_PROMPT_READ_ONLY` is the complete prompt used to establish the research question, conceptual frame, open-coding practice, span rules, and entity/relation style for the existing annotations. Read it as historiographical, conceptual, linguistic, and methodological background. It is not the present task. Do not follow its requirement to annotate every sentence, do not aim for its page-level density target, and do not recreate or replace the existing annotations. Follow only the instructions in this enrichment prompt.

## Purpose of this pass

The existing v2 annotations are generally good, but some long Burmese chronicle sentences were collapsed into one or a few umbrella triples even though the text states several distinct predicates. Re-read each supplied `SENTENCE_RECORD` while treating the `EXISTING_TRIPLES_READ_ONLY` printed directly beneath that sentence as the read-only record of what has already been captured. Return only additional text-stated predicates that those existing triples missed.

The basic task is to record who did what to whom or what, as actually stated in the sentence. You are not being asked to infer or theorize additional power relations from titles, offices, kinship labels, places, entity types, or historical background. The sociology-of-power research question is only the relevance filter for deciding which predicates are worth recording. It is not permission to manufacture a relation that the sentence does not predicate.

Relevant textual predicates include reporting and transmitting information; commanding and delegating action; mobilizing, fighting, suppressing, capturing, appointing, granting, incorporating, serving, giving tribute, transferring resources, sponsoring ritual, consecrating, distributing, worshipping, adjudicating, resisting, and comparable actions or relations actually expressed by the Burmese sentence and clarified by its supplied translation.

The supplied summary is orientation, not an extraction checklist or ceiling. The source sentence and its translation are the evidence. The existing triples are a deduplication reference, not a complete interpretation of the sentence.

## Non-negotiable predicate rule

Every proposed `p` must correspond to a real predicate or relational construction in the assigned sentence. The predicate may be active, passive, causative, reported, or grammatically elliptical, and one endpoint may be omitted and legitimately inferred. But the relation itself must be stated by the sentence. Do not infer a predicate merely because a title or noun phrase implies a historically plausible status.

For example, `မြို့စား` inside a person's title identifies that person as a town lord. By itself it does not state that anyone assigned, granted, or transferred an appanage. Do not return `IS_ASSIGNED_APPANAGE_OF` from a `မြို့စား` title alone. Such an annotation is valid only when the sentence actually predicates appointment, granting, assignment, inheritance, removal, or transfer through wording such as `ဘွားစားစေ`, `မြို့စားပေးတော်မူ`, `ခန့်`, `ပေး`, or an equivalent construction.

Likewise, an office title may help tag a subject, a kinship term may identify an endpoint, and a place name may locate an action, but none of them is automatically a new subject-predicate-object claim.

Before returning any addition, privately complete this test:

1. State the concrete “who did what to whom or what” proposition expressed by the sentence.
2. Identify the Burmese predicate or relational construction, using the English translation only to resolve its meaning.
3. Ask whether the same proposed relation could still be invented if all verbs and relational constructions were removed and only the titles and nouns remained. If yes, reject it.
4. Convert the real textual predicate into a concise analytical `p` label without changing what the sentence says.
5. Compare that proposition against the existing triples and return it only if it is missing.

## Clause-by-clause enrichment audit

Pay special attention to long or compound sentences. Privately audit each sentence clause by clause and ask:

1. What distinct predicates or relational constructions does the sentence actually state?
2. Who performs each predicate, and to whom or what is it directed?
3. Which of those stated predicates are relevant to the research question?
4. Has that same textual proposition already been represented by an existing triple, even if its label uses different words?
5. Which relevant, text-stated predicates remain genuinely missing?

An umbrella triple does not automatically cover every distinct predicate in the sentence. Reporting an attack, ordering a response, mobilizing commanders, capturing leaders, and reporting the outcome are separate relations when the sentence explicitly distinguishes those actions. Likewise, commissioning sacred objects, completing and gilding them, consecrating them, installing them across the realm, naming them, and organizing public worship are not interchangeable merely because they belong to one larger royal religious project.

Add a triple when it captures a materially distinct relation, including when:

- the sentence states a different predicate, not merely another noun, title, attribute, or synonym;
- the acting subject changes;
- the receiving or affected object changes;
- the direction of power or information changes;
- the sentence moves to a distinct institutional or ritual stage;
- a broad existing triple omits a separately stated act that matters to the sociology of power.

Do not add a triple merely to provide:

- a synonymous or slightly more specific label for an existing claim;
- an inferred status, possession, appointment, appanage, authority, or institutional relation derived only from a title or entity description;
- another item from a location, personnel, object, or quantity list governed by the same predicate;
- a date, place, quantity, title, or descriptive modifier with no distinct relation;
- a decomposition of one predicate into several paraphrases;
- a relation that is historically interesting but not supported by the assigned sentence.

It is acceptable to add several triples to one rich sentence and none to every other sentence on the page. There is no quota. Precision against semantic duplication matters, but do not let one broad existing gloss hide several distinct predicates.

## Positive examples: complete first-pass records followed by missing-relation analysis

These examples diagnose the under-annotation failure mode. Each one prints the complete Burmese sentence, its complete supplied English translation, and every existing triple before explaining what is missing. The examples guide the coverage decision; they are not a fixed relation-label vocabulary.

### Positive example 1: `vol3_s000738`

Full Burmese sentence:

`ထိုသက္ကရာဇ် သီတင်းကျွတ်လအတွင်း ကန်ပြင်ရွာနေ ဗိုလ်ခေါ်ခံ ငလန်း, ငမြတ်ထွား၊ ဂန္ဓမာရွာနေ ဗိုလ်ခေါ်ခံ ငဝိုင်း, ငထိုက်၊ ဆင်ရွာကြီးရွာနေ ဗိုလ်ခေါ်ခံ ငမှုံတို့နှင့် လူတစ်ထောင့်ခြောက်ရာကျော်၊ ဆင်ငါးစီး၊ အကြီးပြုလုပ်သူ နတ်စုလက်ယာ သေနတ်စာရေးဟောင်း ငလှောက်တို့ မငြိမ်မသက် လူသူစုရုံးပြီးလျှင် တောင်ပြုံးကြီးမြို့ကို တိုက်ခိုက် ကြောင်းကို အိမ်ရှေ့တော်ထမ်း ငအိ, အမရပူရမြို့စောင့် မိုးညှင်းမြို့စားဝန်ကြီးတို့က လျှောက်ထားကြားသိတော်မူလျှင် အိမ်ရှေ့ဝန် နတ်မောက်မြို့စား မင်းကြီးမဟာကျော်ထင်၊ လက်ယာကြောင်းဗိုလ် မဟာမင်းခေါင်ကျော်ထင်, မတ္တရာ ကျော်စဉ်တိုက်ဝန်တို့ကို အစုအမှုထမ်း ခြောက်ရာကျော်ပေးအပ်၍ အမရပူ ရမြို့က လက်နက်ကိုင်လူတို့ကို ချုပ်အုပ် တိုက်ခိုက် ဖမ်းဆီးရမည် အမိန့်တော်နှင့် ချီတက်စေရာ မိုးညှင်းမြို့စားဝန်ကြီးနှင့် အိမ်ရှေ့ဝန်တို့က အကြီးပြုသူ ငဝိုင်း, ငလှောက်တို့နှင့် သားမယားတို့ကို မိကြောင်း၊ အစုအရုံး မရှိကြောင်းကို လျှောက်ထားပြန်လာရ၏။`

Full English translation:

`During the month of Thadingyut of that year, when the Minister of Moehnyin and the Crown Prince's attendant reported that the so-called leaders Nga Lan and Nga Myat Htwa residing in Kanpyin village, the so-called leaders Nga Waing and Nga Htaik residing in Gandama village, and the so-called leader Nga Hmon residing in Sin-ywa-gyi village, along with over one thousand six hundred people, five elephants, and the former Nat-su-let-yar gun-clerk Nga Hlyauk who acted as the leader, had gathered people in an unstable manner and were attacking Taungbyon-gyi city, [the King] heard the report and ordered the Crown Prince's attendant, the Lord of Natmauk, Minister Maha Kyawhtin, the Right-wing Commander Maha Minhkaung Kyawhtin, and the Mattaya Kyawzin District Minister to take over six hundred service men, proceed from Amarapura city, and suppress, attack, and arrest the armed men; the Minister of Moehnyin and the Crown Prince's attendant returned and reported that they had captured the leaders Nga Waing and Nga Hlyauk along with their wives and children, and that there was no further gathering.`

Existing triples in full:

1. s = `ငလန်း, ငမြတ်ထွား, ငဝိုင်း, ငထိုက်, ငမှုံ, ငလှောက်` | en = “Nga Lan, Nga Myat Htwa, Nga Waing, Nga Htaik, Nga Hmon, Nga Hlyauk” | tag = `RebelLeader` | source = `sentence`; p = `ATTACKS_ROYAL_CENTER`; o = `တောင်ပြုံးကြီးမြို့` | en = “Taungbyon-gyi city” | tag = `RoyalAdministrativeCenter` | source = `sentence`.
2. s = `ဘဝရှင်မင်းတရားကြီးဘုရား` | en = “The Living Lord King” | tag = `RoyalAuthority` | source = `inferred`; p = `MOBILIZES_MILITARY_FORCE_TO_SUPPRESS`; o = `အိမ်ရှေ့ဝန် နတ်မောက်မြို့စား မင်းကြီးမဟာကျော်ထင်, လက်ယာကြောင်းဗိုလ် မဟာမင်းခေါင်ကျော်ထင်, မတ္တရာ ကျော်စဉ်တိုက်ဝန်` | en = “Lord of Natmauk, Minister Maha Kyawhtin, Right-wing Commander Maha Minhkaung Kyawhtin, and Mattaya Kyawzin District Minister” | tag = `MilitaryCommander` | source = `sentence`.
3. s = `အိမ်ရှေ့ဝန် နတ်မောက်မြို့စား မင်းကြီးမဟာကျော်ထင်, လက်ယာကြောင်းဗိုလ် မဟာမင်းခေါင်ကျော်ထင်, မတ္တရာ ကျော်စဉ်တိုက်ဝန်` | en = “Lord of Natmauk, Minister Maha Kyawhtin, Right-wing Commander Maha Minhkaung Kyawhtin, and Mattaya Kyawzin District Minister” | tag = `MilitaryCommander` | source = `sentence`; p = `CAPTURES_REBEL_LEADERS_AND_FAMILIES`; o = `ငဝိုင်း, ငလှောက်တို့နှင့် သားမယား` | en = “Nga Waing, Nga Hlyauk and their wives and children” | tag = `RebelLeader` | source = `sentence`.

New triple required for exhaustive coverage:

Manual pointer: “Moehnyin and the Crown Prince's attendant reported the outbreak of rebellion.”

1. s = `အိမ်ရှေ့တော်ထမ်း ငအိ, အမရပူရမြို့စောင့် မိုးညှင်းမြို့စားဝန်ကြီးတို့` | en = “Nga Ei, attendant of the Crown Prince, and the Minister of Moehnyin guarding Amarapura” | tag = `RoyalReportingOfficials` | source = `sentence`; p = `REPORTS_REBEL_MOBILIZATION_AND_ATTACK_TO`; o = `ဘဝရှင်မင်းတရားကြီးဘုရား` | en = “The Living Lord King” | tag = `RoyalAuthority` | source = `page`.

The other manual pointers—the rebels attacking Taungbyon-gyi, the king ordering a military expedition, and the capture of the rebel leaders and their families—are already represented by the displayed existing triples. Do not return them again.

### Positive example 2: `vol3_s000745`

Full Burmese sentence:

`ထိုသက္ကရာဇ်အတွင်း ရတနာသိင်္ဃ မြို့တည်နန်းတည် ဘေးတော် အလောင်းမင်းတရားကြီးဘုရား မြေနန်း ပြာသာဒ်တော် တိုင်သစ်များကို ထုလုပ်တော်မူသည့် ဘုရားရုပ်တုဆူရေ တစ်ရာ့ငါး, ရဟန္တာရုပ်တု ဆူရေ ကိုးဆယ့်ရှစ်, နှစ်စုဆူရေ နှစ်ရာ့သုံး ထုလုပ် သစ်စေး သားရိုး ရွှေမွမ်းမံ ပြီးပြေသည်ကို စံနန်းတော်ရှေ့တန်ဆောင်းတွင် အနေကဇာ တင်၍ ကိန်းဝပ်စေပြီးလျှင် ရတနာသိင်္ဃမြို့ တော်အတွင်း စွယ်တော်စင်အနီးတွင် တစ်ဆူ၊ မင်းကျောင်းရွာတွင် တစ်ဆူ၊ ဆီးတောရွာတွင် တစ်ဆူ၊ ချည်ပါရွာတွင် တစ်ဆူ၊ ထနောင်းဝန်းရွာတွင် တစ်ဆူ၊ ဆိတ်ခွန်ရွာတွင် တစ်ဆူ၊ သီလုံးရွာတွင် တစ်ဆူ၊ ကြိုးကြာရွာတွင် တစ်ဆူ၊ အုံးပေါက်ရွာတွင် တစ်ဆူ၊ ပလိုင်းရွာတွင် တစ်ဆူ၊ သင်ပန်းချောင်းရွာတွင် တစ်ဆူ၊ ယင်းတောရွာတွင် တစ်ဆူ၊ စိုင်နံတောင်ရွာတွင် တစ်ဆူ၊ မြောက်ရွာတွင် တစ်ဆူ၊ ဟန်လင်းမြို့တွင် တစ်ဆူ၊ ကျားရွာတွင် တစ်ဆူ၊ မောက်ကျိုးရွာတွင် တစ်ဆူ၊ ကော့ရွာတွင် တစ်ဆူ၊ မှတ်ထိရွာတွင် တစ်ဆူ၊ လှတောကြီးငါးရွာတွင် တစ်ဆူ၊ မူးသာရွာတွင် တစ်ဆူ၊ ကန်သာရွာတွင် တစ်ဆူ၊ လက်ပံလှရွာတွင် တစ်ဆူ၊ သရိုင်ရွာတွင် တစ်ဆူ၊ မန်ကျည်းတုံရွာတွင် တစ်ဆူ၊ ဆင်အင်းရွာတွင် တစ်ဆူ၊ ယင်းပါရွာတွင် တစ်ဆူ၊ အင်းဘဲရွာတွင် တစ်ဆူ၊ ကြေးကတူးရွာတွင် တစ်ဆူ၊ သရစိမ်းရွာတွင် တစ်ဆူ၊ သမန်သာရွာတွင် တစ်ဆူ၊ ဆားထောင် မူးသာရွာတွင် တစ်ဆူ၊ ညောင်ပင်သာရွာတွင် တစ်ဆူ၊ စည်ပုတ္တရာရွာတွင် တစ်ဆူ၊ ပေကုန်းရွာတွင် တစ်ဆူ၊ ညောင်စဉ်သင်ပေါင်းကျဉ်းရွာတွင် တစ်ဆူ၊ သာဝတ္ထိရွာတွင် တစ်ဆူ၊ ယင်းတိုက်ရွာတွင် တစ်ဆူ၊ ခေါတောရွာတွင် တစ်ဆူ၊ မြင်းသည်ရွာတွင် တစ်ဆူ၊ ရုံးသာရွာတွင် တစ်ဆူ၊ ခင်ဦးရွာတွင် တစ်ဆူ၊ ပြွန်ဦးရွာတွင် တစ်ဆူ၊ ရွာသာရွာတွင် တစ်ဆူ၊ လှထွေရွာတွင် တစ်ဆူ၊ ရွာဝေးရွာတွင် တစ်ဆူ၊ စည်သာမြို့ တွင် တစ်ဆူ၊ ကူခေါင်းတောင်ရွာတွင် တစ်ဆူ၊ တောင်ကြားရွာတွင် တစ်ဆူ၊ ငစဉ့်ကိုင်ရွာတွင် တစ်ဆူ၊ မြကန်ရွာတွင် တစ်ဆူ၊ ဆင်းကွပ်ရွာတွင် တစ်ဆူ၊ တလိုင်း ရွာတွင် တစ်ဆူ၊ ဝက်လည်ရွဲရွာတွင် တစ်ဆူ၊ ပန်းတံဆိပ်ရွာတွင် တစ်ဆူ၊ ဥနှဲဘုတ်ရွာတွင် တစ်ဆူ၊ သရက်ကန်ရွာတွင် တစ်ဆူ၊ မန်သာရွာတွင် တစ်ဆူ၊ စောကြီးရွာတွင် တစ်ဆူ၊ ရွာနန်းရွာတွင် တစ်ဆူ၊ ဒန့်ကျည်းရွာတွင် တစ်ဆူ၊ ပဲကူးတောင်ရွာတွင် တစ်ဆူ၊ ပဲကူးမြောက်ရွာတွင် တစ်ဆူ၊ ဆင်ပြာကုန်းရွာတွင် တစ်ဆူ၊ မြောင်တောင်ရွာတွင် တစ်ဆူ၊ မြောင်မြောက်ရွာတွင် တစ်ဆူ၊ စုစုဆူရေ ခြောက်ဆယ့်ငါးများကို လောကမာရဇိန်နှင်နှင် ကမ္ပည်းတပ်ပြီးလျှင် သက္ကရာဇ် (၁၂၁၅) တစ်ထောင့်နှစ်ရာ့တစ်ဆယ့်ငါးခု တန်ဆောင်မုန်းလဆန်းတစ်ရက် နေ့က စ၍ အသီးအသီး ယောက်ျားမိန်းမ အကပွဲသဘင်များနှင့် ထီးဖြူစိုက်ဆောက်သော ဆင်ကထက် တင်ဆောင်ကြိုယူကိုးကွယ်စေ၏။`

Full English translation:

`During that year, the 105 Buddha statues and 98 Arahant statues, for which the great-grandfather, the founder of the city and palace, King Alaungmintaya, had carved the new pillars for the royal palace pavilion, were completed with carving, lacquer, and gold gilding; after holding an Anet-ka-za ceremony in the pavilion in front of the royal palace to consecrate them, [the King] had them installed, with one in the vicinity of the Tooth Relic shrine in Ratanasingha city, one in Min-kyaung village, one in Hsi-taw village, one in Chi-par village, one in Htan-naw-wun village, one in Seit-khun village, one in Thi-lon village, one in Kyo-kya village, one in On-pauk village, one in Pa-line village, one in Thin-pan-chaung village, one in Yin-taw village, one in Saing-nan-taung village, one in Myauk village, one in Hanlin city, one in Kya village, one in Mauk-kyo village, one in Kaw village, one in Hmat-hti village, one in the five villages of Hla-taw-gyi, one in Mu-tha village, one in Kan-tha village, one in Let-pan-hla village, one in Tha-yaing village, one in Man-kyi-ton village, one in Sin-in village, one in Yin-par village, one in In-be village, one in Kye-ka-tu village, one in Tha-sein village, one in Tha-man-tha village, one in Sar-htaung Mu-tha village, one in Nyaung-pin-tha village, one in Si-put-ta-ya village, one in Pe-kon village, one in Nyaung-zin Thin-paung-kyin village, one in Thawatti village, one in Yin-taik village, one in Hkaw-taw village, one in Myin-thi village, one in Yon-tha village, one in Khin-u village, one in Pyun-u village, one in Ywa-tha village, one in Hla-htwe village, one in Ywa-we village, one in Si-tha city, one in Ku-hkawng-taung village, one in Taung-kya village, one in Nga-zin-kine village, one in Mya-kan village, one in Sin-kut village, one in Ta-laing village, one in Wet-let-ywe village, one in Pan-teik-seik village, one in U-hne-bote village, one in Tha-yet-kan village, one in Man-tha village, one in Saw-gyi village, one in Ywa-nan village, one in Dan-kyi village, one in Pe-ku-taung village, one in Pe-ku-myauk village, one in Sin-pya-kon village, one in Myaung-taung village, and one in Myaung-myauk village, for a total of sixty-five statues, which were inscribed with the name 'Lokamarazein' and, starting from the first day of the waxing moon of Tazaungmon in the year 1215, were taken to be worshipped by the people, men and women, with dance and music festivals and carried on elephant-back under white umbrellas.`

Existing triples in full:

1. s = `ဘဝရှင်မင်းတရားကြီးဘုရား` | en = “The Living Lord King” | tag = `RoyalAuthority` | source = `inferred`; p = `DISTRIBUTES_SACRED_OBJECTS_TO_LOCAL_COMMUNITIES`; o = `ဘုရားရုပ်တု` | en = “Buddha statues” | tag = `SacredObject` | source = `sentence`.

New triples required for exhaustive coverage:

Manual pointer: “Alaungmintaya carved the palace pillars for the Buddha and Arahant statues.”

1. s = `ဘေးတော် အလောင်းမင်းတရားကြီးဘုရား` | en = “the royal ancestor King Alaungmintaya” | tag = `DynasticRoyalAncestor` | source = `sentence`; p = `FASHIONS_PALACE_PILLARS_INTO_SACRED_IMAGES`; o = `ဘုရားရုပ်တုဆူရေ တစ်ရာ့ငါး, ရဟန္တာရုပ်တု ဆူရေ ကိုးဆယ့်ရှစ်` | en = “105 Buddha images and 98 Arahant images” | tag = `RoyalSacredImageCorpus` | source = `sentence`.

Manual pointer: “The present king commissioned the completion of the lavish statues.”

2. s = `ဘဝရှင်မင်းတရားကြီးဘုရား` | en = “The Living Lord King” | tag = `RoyalAuthority` | source = `page`; p = `COMMISSIONS_COMPLETION_LACQUERING_AND_GILDING_OF`; o = `ဘုရားရုပ်တုဆူရေ တစ်ရာ့ငါး, ရဟန္တာရုပ်တု ဆူရေ ကိုးဆယ့်ရှစ်` | en = “105 Buddha images and 98 Arahant images” | tag = `RoyalSacredImageCorpus` | source = `sentence`.

Manual pointer: “A ceremony was held to consecrate the statues.”

3. s = `ဘဝရှင်မင်းတရားကြီးဘုရား` | en = “The Living Lord King” | tag = `RoyalAuthority` | source = `page`; p = `CONSECRATES_AND_ENSHRINES_SACRED_IMAGES_AT_PALACE`; o = `ဘုရားရုပ်တုဆူရေ တစ်ရာ့ငါး, ရဟန္တာရုပ်တု ဆူရေ ကိုးဆယ့်ရှစ်` | en = “105 Buddha images and 98 Arahant images” | tag = `RoyalSacredImageCorpus` | source = `sentence`.

Manual pointer: “The statues were worshipped by local communities.”

4. s = `အသီးအသီး ယောက်ျားမိန်းမ` | en = “men and women of the various local communities” | tag = `LocalDevotionalCommunities` | source = `sentence`; p = `CEREMONIALLY_TRANSPORTS_AND_WORSHIPS`; o = `စုစုဆူရေ ခြောက်ဆယ့်ငါးများ` | en = “the total of sixty-five sacred images” | tag = `DistributedSacredImages` | source = `sentence`.

The remaining manual pointer—the installation of the statues in locations throughout the realm—is already represented by `DISTRIBUTES_SACRED_OBJECTS_TO_LOCAL_COMMUNITIES`. Do not return one triple per place, and do not repeat that existing distribution relation.

### Positive example 3: Ceylonese ordination dispute and Sacred Tooth diplomatic mission

Full Burmese sentence:

`ထိုသက္ကရာဇ်အတွင်း အထက်လာရောက်ရင်း သီဟိုဠ်ကျွန်း ကဏ္ဏတိတ္ထမြို့ နေ ရဟန်းတော် ရှင်ဓမ္မက္ခန္ဓ, ရှင်ဝဏ္ဏာရတနတို့ သီဟိုဠ်ရဟန်းတို့ သိမ်အယူဝါဒ ကွဲပြားသည့်အမှုနှင့် မန္တလေးရတနာပုံ ရွှေမြို့တော်သို့ လာရောက် ပြန်၍ သာသနာပြုဆရာတော်ကြီးထံ လျှောက်ထားရာ သီမဝိနိစ္ဆယ အဆုံးအဖြတ်ကို ပေးအပ်ဆောင်ယူ ပြန်သွား မည် ရှိသည်တွင် သီဟိုဠ်ရှိ စွယ်တော်မြတ်ကို ပင့်ဆောင်ကိုးကွယ်လိုသည့် အကြောင်းကို သန္ဓေသကထာစကား ရေးသား၍ သီဟိုဠ်ကျွန်းရှိ ထေရ်ကြီးဝါကြီး ရဟန်းတို့ကို လှူဒါန်းရန် ပရိက္ခရာ, သီဟိုဠ်ကျွန်းကို ချုပ်အုပ်သော သူတို့အား ပေးကမ်းရန်များနှင့် သန္ဓေသကထာစာများကို ကျွန်တော်ရင်း အတွင်းတော်မြဲ ငကြီး, သီလသမာဓိရှိသူ ဥပုသ်တော်ခေါင်း ငမွှေး, ငအောင်ထွန်းတို့သို့ ပေးအပ်ပြီးလျှင် ရှင်ဓမ္မက္ခန္ဓ, ရှင်ဝဏ္ဏာရတနတို့နှင့်အတူ ဖက်ထည့် စေလွှတ်တော်မူသည်။`

Full English translation:

`During that year, the monks Shin Dhammakkhandha and Shin Wannaratana, who were originally from Kanthittha City in the island of Ceylon and had come to the great golden city of Mandalay Yadanabon regarding the matter of the difference in the ordination doctrines of the Ceylonese monks, returned and submitted the matter to the great presiding monk of the Sasana; while they were to be given the decision of the Sima-vinicchaya to take back, [the King] wrote a letter of proposal regarding the desire to invite and worship the Sacred Tooth Relic from Ceylon, and after entrusting the requisites to be donated to the senior and elder monks of Ceylon, the items to be given to those who govern the island of Ceylon, and the letters of proposal to his own inner royal attendants Nga Gyi, and the virtuous head of the Sabbath-keepers Nga Mway and Nga Aung Htun, he sent them along with Shin Dhammakkhandha and Shin Wannaratana.`

Existing triples in full:

1. s = `မင်း` | en = “the King” | tag = `RoyalAuthority` | source = `inferred`; p = `DISPATCHES_DIPLOMATIC_MISSION`; o = `ကျွန်တော်ရင်း အတွင်းတော်မြဲ ငကြီး, သီလသမာဓိရှိသူ ဥပုသ်တော်ခေါင်း ငမွှေး, ငအောင်ထွန်း` | en = “inner royal attendants Nga Gyi, Nga Mway, Nga Aung Htun” | tag = `RoyalDiplomaticAgents` | source = `sentence`.
2. s = `မင်း` | en = “the King” | tag = `RoyalAuthority` | source = `inferred`; p = `REQUESTS_SACRED_RELIC_FROM`; o = `သီဟိုဠ်ကျွန်း` | en = “island of Ceylon” | tag = `ForeignBuddhistCenter` | source = `sentence`.
3. s = `မင်း` | en = “the King” | tag = `RoyalAuthority` | source = `inferred`; p = `PATRONIZES_FOREIGN_SANGHA`; o = `သီဟိုဠ်ကျွန်းရှိ ထေရ်ကြီးဝါကြီး ရဟန်းတို့` | en = “senior and elder monks of Ceylon” | tag = `ForeignReligiousElite` | source = `sentence`.

New triples required for exhaustive coverage:

Manual pointer: “Sri Lankan monks travel to Mandalay in order to discuss the Ceylonese ordination-doctrine dispute.”

1. s = `ရှင်ဓမ္မက္ခန္ဓ, ရှင်ဝဏ္ဏာရတနတို့` | en = “Shin Dhammakkhandha and Shin Wannaratana” | tag = `ForeignMonasticPetitioners` | source = `sentence`; p = `TRAVELS_TO_MANDALAY_TO_SUBMIT_ORDINATION_DISPUTE_TO`; o = `သာသနာပြုဆရာတော်ကြီး` | en = “the great presiding monk of the Sasana” | tag = `BurmeseMonasticAuthority` | source = `sentence`.

Manual pointer: “The head Burmese monk is represented as deciding or governing Buddhist affairs in Sri Lanka.”

2. s = `သာသနာပြုဆရာတော်ကြီး` | en = “the great presiding monk of the Sasana” | tag = `BurmeseMonasticAuthority` | source = `sentence`; p = `ISSUES_BINDING_SIMA_RULING_FOR`; o = `သီဟိုဠ်ရဟန်းတို့ သိမ်အယူဝါဒ ကွဲပြားသည့်အမှု` | en = “the Ceylonese monks' dispute over ordination doctrine” | tag = `TransregionalOrdinationDispute` | source = `sentence`.

Manual pointer: “The king entrusts donations for the elder monks and rulers of Ceylon, together with proposal letters, to his diplomats.”

3. s = `ဘဝရှင်မင်းတရားကြီးဘုရား` | en = “The Living Lord King” | tag = `RoyalAuthority` | source = `page`; p = `SENDS_DIPLOMATIC_GIFTS_AND_PROPOSAL_LETTERS_TO`; o = `သီဟိုဠ်ကျွန်းကို ချုပ်အုပ်သော သူတို့` | en = “those who govern the island of Ceylon” | tag = `ForeignGoverningAuthorities` | source = `sentence`.

The other manual pointers—the king writing a proposal to retrieve the Sacred Tooth Relic, patronizing the elder monks of Ceylon, and dispatching the diplomatic mission—are already represented by the displayed existing triples. Do not return them again.

## Negative examples: complete first-pass records with no missing relations

### Negative example 1: gate names and commemorative pillars

Full Burmese sentence:

`တံခါးတော်ပြင် မီးတားရွက်ကာအနီး အုတ်ပလ္လင်သုံးဆင့်ခံ၍ မော်ကွန်းတိုင်စိုက်ထူ ကမ္ပည်းရွက်ကပ်ပြီးလျှင် သက္ကရာဇ် (၁၂၂၁) တစ်ထောင့်နှစ်ရာ့နှစ်ဆယ့်တစ်ခု ကဆုန်လပြည့်ကျော် ခြောက်ရက် ညဉ့်သုံးချက်တီးကျော် ခုနစ်ရက် တနင်္လာနေ့သို့အဝင် လေးနာရီ နှစ်ပါဒ် အချိန်တည် ရတနာပုံရွှေမြို့တော်ကြီး ဦးထိပ်တံခါး, သောင်းညွတ် တံခါး စသော ဆိုင်ရာ တံခါးအမည်နှင့်တကွ တံခါးတစ်ဆယ့်နှစ်ရပ် စေ့ငုအောင် ကမ္ပည်းတပ်မှတ်ရ၏။`

Full English translation:

`Outside the royal gates, near the fire-prevention screens, three-tiered brick pedestals were built, commemorative pillars were erected, and inscriptions were affixed; then, at the time of 4 hours and 2 parts [of the night] entering Monday, the 6th day after the full moon of Kason, 1221 B.E., past the third strike of the night, the twelve gates of the great golden city of Yadanabon, including the U-hteik Gate and the Thaung-nyut Gate, were marked with their respective names.`

Existing triples in full:

1. s = `မင်း` | en = “the King” | tag = `RoyalAuthority` | source = `inferred`; p = `NAMES_AND_LEGITIMATES_CAPITAL_GATES`; o = `တံခါးတစ်ဆယ့်နှစ်ရပ်` | en = “the twelve gates” | tag = `CapitalInfrastructure` | source = `sentence`.
2. s = `မင်း` | en = “the King” | tag = `RoyalAuthority` | source = `inferred`; p = `ERECTS_COMMEMORATIVE_MONUMENTS`; o = `မော်ကွန်းတိုင်` | en = “commemorative pillars” | tag = `RoyalSacredArchitecture` | source = `sentence`.

### Negative example 2: city-wall measurements and Buddhist time

Full Burmese sentence:

`မြို့ရိုးအုတ်မြစ်သံ တစ်တောင်၊ အုတ်ရိုးထုတစ်တာ၊ အရပ် သံတစ်ဆယ့်ငါးတောင်၊ သူရဲခိုသံ သုံးတောင်၊ နှစ်စုအရပ် တစ်ဆယ့်ရှစ်တောင်၊ အချင်းတာ ခြောက်ရာ၊ လေးမျက်နှာတာပေါင်း နှစ်ထောင့်လေးရာ၊ ဤကဲ့သို့ တည်လုပ်သည်မှာ သာသနာတော်နှစ် (၂၄၀၀) နှစ်ထောင့်လေးရာ ရောက်သောအခါ၌ တည်လုပ်ရသည် ဖြစ်သော ကြောင့်တည်း။`

Full English translation:

`The city wall foundation is one taung thick, the wall thickness is one ta, the height is fifteen taung, the parapet height is three taung, the height of the two sections is eighteen taung, the diameter is six hundred ta, and the total for the four sides is two thousand four hundred ta; this construction was carried out because it was the time when the year 2400 of the Sasana had arrived.`

Existing triples in full:

1. s = `မင်း` | en = “the King” | tag = `RoyalAuthority` | source = `inferred`; p = `CONSTRUCTS_CAPITAL_DEFENSES`; o = `မြို့ရိုး` | en = “city wall” | tag = `CapitalInfrastructure` | source = `sentence`.
2. s = `သာသနာတော်နှစ် (၂၄၀၀)` | en = “year 2400 of the Sasana” | tag = `BuddhistCosmologicalTime` | source = `sentence`; p = `LEGITIMATES_ROYAL_CONSTRUCTION`; o = `မြို့တည်လုပ်ခြင်း` | en = “city construction” | tag = `RoyalStateBuilding` | source = `inferred`.

### Negative example 3: Kengtung Sawbwa sends a precious gem

Full Burmese sentence:

`ကျိုင်းရုံးကြီးစော်ဘွားကလည်း ကျိုင်းရုံးကြီးမြို့ တွင် ထူးမြတ်သည့် ရတနာကျောက်တော် ရသည်ကို မြို့ စစ်ကဲ မင်းထင်ရဲလှကျော်သူ, အမတ်ဘယားစန္ဒရံ, ဘယားစုံစံတို့ ကြီးကြပ်ပို့ဆက်လာသည်ကို အစဉ်ထုံးစံ ရှိသည့်အတိုင်း ကြိုယူမြဲ ကြိုယူ ဆက်သွင်းရ၏။`

Full English translation:

`The Kengtung Sawbwa also had a precious gem of great merit found in Kengtung town supervised and delivered by the town Sitke Minhtin Ye Hla Kyaw Thu, the minister Bayasandara, and Bayasonsan, and it was received and submitted as was the customary tradition.`

Existing triples in full:

1. s = `ကျိုင်းရုံးကြီးစော်ဘွား` | en = “Kengtung Sawbwa” | tag = `FrontierSubmission` | source = `sentence`; p = `SENDS_TRIBUTE_TO`; o = `ဘဝရှင် မင်းတရားကြီးဘုရား` | en = “Sovereign King” | tag = `FrontierSubmission` | source = `inferred`.
2. s = `မြို့ စစ်ကဲ မင်းထင်ရဲလှကျော်သူ, အမတ်ဘယားစန္ဒရံ, ဘယားစုံစံ` | en = “town Sitke Minhtin Ye Hla Kyaw Thu, minister Bayasandara, and Bayasonsan” | tag = `AdministrativeTributeDelivery` | source = `sentence`; p = `DELIVER`; o = `ရတနာကျောက်တော်` | en = “precious gem” | tag = `SacredObjectCentralization` | source = `sentence`.

There are no additional relations of note in any of these three negative examples. The first pass was exhaustive for these sentences. Return no additions for records like these merely because further grammatical predicates, measurements, dates, provenance details, intermediary steps, protocol formulas, or list items could be restated as triples.

## Output and grounding rules

Return the same sentence-grouped triple structure used by the existing v2 annotations:

- `S`: a list containing only sentences with one or more genuinely new triples. Return `S: []` if the page needs no additions.
- `sid`: the exact supplied sentence ID.
- `T`: one or more additional triples supported by that sentence.
- `s` and `o`: each contains `my`, `en`, `tag`, and `source`, following the meanings established by the read-only prompt.
- `p`: a concise analytical graph-edge label in `ALL_CAPS_WITH_UNDERSCORES`.

Preserve supplied sentence order among the `S` records you do return, and proposition order within each `T` list. Do not return a sentence record with an empty `T` list.

Every new triple must be supported by its assigned sentence. When a subject or object is explicit in that sentence, use an exact Burmese span and set `source` to `sentence`. Use `page` only when the endpoint is explicit elsewhere among the supplied page sentences. Use `inferred` only when the endpoint is genuinely absent from all supplied Burmese sentences and must be reconstructed from grammar or chronicle convention. Prefer explicit endpoints when available.

Do not modify, replace, critique, score, or repeat existing triples in the response. Do not return a new summary, translations, explanations, coverage report, confidence field, evidence field, metadata modifiers, or deletion instructions. Return only schema-valid JSON through the supplied response schema.
