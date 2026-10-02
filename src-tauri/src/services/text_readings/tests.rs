//! Plugin-level tests of `text_readings` (P6-L): wire format, request handling, limits, and the
//! reading fixtures (Chinese through the real jieba + pinyin data; Japanese from recorded NJD
//! features, plus the real dictionary when `SIGIL_TEST_MODELS_DIR` has it).

use super::*;

/// The jpreprocess NAIST-JDIC folder for the real-dictionary tests:
/// `$SIGIL_TEST_MODELS_DIR/naist-jdic` (from jpreprocess v0.15.0's `naist-jdic-jpreprocess.tar.gz`).
fn test_dictionary() -> Option<std::path::PathBuf> {
    let root = std::env::var_os("SIGIL_TEST_MODELS_DIR")?;
    japanese::dictionary_dir(&std::path::PathBuf::from(root).join("naist-jdic"))
}

#[test]
#[ignore = "development helper: prints NJD features and readings of SIGIL_READINGS_DUMP ('|'-separated; 'zh:' prefix for Chinese)"]
fn dump_readings() {
    let texts = std::env::var("SIGIL_READINGS_DUMP").unwrap_or_else(|_| "私は学生です。|zh:你好世界".into());
    let analyzer = test_dictionary().map(|dir| japanese::JapaneseAnalyzer::load(&dir).unwrap());
    for text in texts.split('|') {
        if let Some(zh_text) = text.strip_prefix("zh:") {
            let r = zh(zh_text, PinyinStyle::Marks);
            println!("== {zh_text} {:?}
{}", chinese_reader().tokens(zh_text), r.pinyin.unwrap_or_default());
            continue;
        }
        let Some(analyzer) = &analyzer else { continue };
        println!("== {text}");
        for f in analyzer.features(text).unwrap() {
            println!("{f}");
        }
        let r = japanese::read(text, &analyzer.morphs(text).unwrap(), true, true);
        println!("{}
{}
{}", r.hiragana.unwrap_or_default(), r.romaji.unwrap_or_default(), segments_string(&finish_segments(r.segments)));
    }
}

// ---------------------------------------------------------------------------------------------
// Chinese (real jieba + pinyin data)
// ---------------------------------------------------------------------------------------------

fn zh(text: &str, style: PinyinStyle) -> ReadingsResult {
    annotate(&ReadingsRequest {
        text: text.into(),
        lang: "zh".into(),
        pinyin: true,
        pinyin_style: style,
        ..ReadingsRequest::default()
    })
    .expect("Chinese readings")
}

fn pinyin_of(text: &str) -> String {
    zh(text, PinyinStyle::Marks).pinyin.unwrap()
}

#[test]
fn chinese_fixtures_read_like_a_dictionary() {
    let cases = [
        // Polyphones and neutral tones from the phrase table.
        ("我们去银行。", "wǒ men qù yín háng."),
        ("他长大了。", "tā zhǎng dà le."),
        ("这个东西很便宜。", "zhè ge dōng xi hěn pián yi."),
        ("重新开始", "chóng xīn kāi shǐ"),
        ("音乐很好听", "yīn yuè hěn hǎo tīng"),
        ("绿色", "lǜ sè"),
        // 不 / 一 sandhi.
        ("我不是学生。", "wǒ bú shì xué sheng."),
        ("一定要来。", "yí dìng yào lái."),
        ("一起去吧。", "yì qǐ qù ba."),
        ("第一", "dì yī"),
        ("十一", "shí yī"),
        ("一百", "yì bǎi"),
        ("你是不是中国人？", "nǐ shì bu shì zhōng guó rén?"),
        ("看一看", "kàn yi kàn"),
        ("好不好", "hǎo bu hǎo"),
        ("要不要", "yào bu yào"),
        ("来不来", "lái bu lái"),
        // A word said twice is not X一X / X不X: both copies keep their sandhi.
        ("不要不要", "bú yào bú yào"),
        ("不对不对", "bú duì bú duì"),
        ("不是不是", "bú shì bú shì"),
        ("一步一步来", "yí bù yí bù lái"),
        ("一天一天过去", "yì tiān yì tiān guò qù"),
        ("一个一个来", "yí ge yí ge lái"),
        ("不不不", "bù bù bù"),
        // Context rules for one-character words.
        ("他跑得很快。", "tā pǎo de hěn kuài."),
        ("我得走了。", "wǒ děi zǒu le."),
        ("慢慢地走", "màn màn de zǒu"),
        // 地: the particle after an adverb or before a verb, the noun dì elsewhere and before 是 / 有.
        ("他认真地学习", "tā rèn zhēn de xué xí"),
        ("孩子们开心地笑了", "hái zi men kāi xīn de xiào le"),
        ("他大声地喊", "tā dà shēng de hǎn"),
        ("她静静地坐着", "tā jìng jìng de zuò zhe"),
        ("这块地是我的。", "zhè kuài dì shì wǒ de."),
        ("这块地很大", "zhè kuài dì hěn dà"),
        ("他把地扫干净了", "tā bǎ dì sǎo gān jing le"),
        ("地有多大", "dì yǒu duō dà"),
        // 教 on its own is the verb jiāo; the noun jiào comes from jieba's words.
        ("她教数学。", "tā jiāo shù xué."),
        ("老师教中文", "lǎo shī jiāo zhōng wén"),
        ("他教过我", "tā jiāo guo wǒ"),
        ("她教我们唱歌", "tā jiāo wǒ men chàng gē"),
        ("教室", "jiào shì"),
        ("宗教", "zōng jiào"),
        // V不了 / V得了: liǎo.
        ("不得了", "bù dé liǎo"),
        ("大不了", "dà bu liǎo"),
        ("受得了", "shòu de liǎo"),
        ("走不了", "zǒu bu liǎo"),
        // Erhua joins its syllable.
        ("一会儿见", "yí huìr jiàn"),
        // Third tone within a word.
        ("可以", "ké yǐ"),
        // Other text is kept.
        ("我用VRChat玩。", "wǒ yòng VRChat wán."),
        // jieba without its HMM keeps 人 / 长 apart, so 长得 reads zhǎng de.
        ("这个人长得很帅。", "zhè ge rén zhǎng de hěn shuài."),
        ("头发很长", "tóu fa hěn cháng"),
        // Measure word, V得C, V不C and the experienced aspect inside jieba words.
        ("我想买一只猫。", "wǒ xiǎng mǎi yì zhī māo."),
        ("说得对", "shuō de duì"),
        ("不好意思，我听不懂。", "bù hǎo yì si, wǒ tīng bu dǒng."),
        ("我去过北京", "wǒ qù guo běi jīng"),
        // 你好 is one jieba word: third-tone sandhi applies.
        ("你好", "ní hǎo"),
    ];
    let mut failures = Vec::new();
    for (text, want) in cases {
        let got = pinyin_of(text);
        if got != want {
            failures.push(format!("{text}: got {got:?}, want {want:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn chinese_styles_and_segments() {
    let r = zh("银行", PinyinStyle::Numbers);
    assert_eq!(r.lang, "zh");
    assert_eq!(r.pinyin.as_deref(), Some("yin2 hang2"));
    assert_eq!(
        r.segments,
        vec![
            ReadingSegment { text: "银".into(), reading: Some("yin2".into()) },
            ReadingSegment { text: "行".into(), reading: Some("hang2".into()) },
        ]
    );
    assert_eq!(zh("银行", PinyinStyle::None).pinyin.as_deref(), Some("yin hang"));
    assert_eq!(zh("一会儿", PinyinStyle::Numbers).pinyin.as_deref(), Some("yi2 hui4r"));
    assert_eq!(zh("吗", PinyinStyle::Numbers).pinyin.as_deref(), Some("ma5"));
    // Segments spell the whole text; non-Han runs carry no reading.
    let r = zh("我用VRChat玩，好吗？", PinyinStyle::Marks);
    let spelled: String = r.segments.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(spelled, "我用VRChat玩，好吗？");
    assert!(r.segments.iter().any(|s| s.text.contains("VRChat") && s.reading.is_none()));
    assert!(r.hiragana.is_none() && r.romaji.is_none());
}

#[test]
fn traditional_and_cantonese_text_read_as_mandarin() {
    let r = annotate(&ReadingsRequest {
        text: "銀行".into(),
        lang: "yue".into(),
        pinyin: true,
        ..ReadingsRequest::default()
    })
    .unwrap();
    assert_eq!(r.lang, "zh");
    // Per character outside the (simplified) phrase table.
    assert!(r.pinyin.as_deref().is_some_and(|p| p.starts_with("yín ")), "{r:?}");
}

// ---------------------------------------------------------------------------------------------
// Requests, limits, wire format
// ---------------------------------------------------------------------------------------------

#[test]
fn languages_parse_like_the_contract() {
    for (lang, want) in [
        ("ja", Some(ReadingLang::Japanese)),
        ("ja-JP", Some(ReadingLang::Japanese)),
        ("JPN", Some(ReadingLang::Japanese)),
        ("zh", Some(ReadingLang::Chinese)),
        ("zh-Hant", Some(ReadingLang::Chinese)),
        ("zh_TW", Some(ReadingLang::Chinese)),
        ("cmn", Some(ReadingLang::Chinese)),
        ("yue", Some(ReadingLang::Chinese)),
        ("en", None),
        ("ko", None),
        ("", None),
    ] {
        assert_eq!(ReadingLang::parse(lang), want, "{lang}");
    }
}

#[test]
fn requests_are_validated() {
    let e = annotate(&ReadingsRequest {
        text: "hello".into(),
        lang: "en".into(),
        romaji: true,
        ..ReadingsRequest::default()
    })
    .unwrap_err();
    assert!(e.contains("'en'"), "{e}");
    let e = annotate(&ReadingsRequest {
        text: "你".repeat(MAX_TEXT_CHARS + 1),
        lang: "zh".into(),
        pinyin: true,
        ..ReadingsRequest::default()
    })
    .unwrap_err();
    assert!(e.contains("too long"), "{e}");
    // Exactly at the limit is fine, and every output stays within the event limits.
    let r = zh(&"中".repeat(MAX_TEXT_CHARS), PinyinStyle::Marks);
    assert!(r.pinyin.as_ref().unwrap().encode_utf16().count() <= MAX_READING_UNITS);
    assert!(r.pinyin.as_ref().unwrap().ends_with('…'));
    assert!(r.segments.len() <= MAX_SEGMENTS);
    let spelled: String = r.segments.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(spelled, "中".repeat(MAX_TEXT_CHARS));
}

#[test]
fn nothing_requested_or_empty_text_needs_no_dictionary() {
    // No Japanese form: no dictionary access, no forms.
    let r = annotate(&ReadingsRequest {
        text: "東京".into(),
        lang: "ja".into(),
        ..ReadingsRequest::default()
    })
    .unwrap();
    assert_eq!((r.lang.as_str(), r.hiragana, r.romaji, r.segments.len()), ("ja", None, None, 0));
    // Empty text: empty forms.
    let r = annotate(&ReadingsRequest {
        text: "  ".into(),
        lang: "ja".into(),
        hiragana: true,
        ..ReadingsRequest::default()
    })
    .unwrap();
    assert_eq!(r.hiragana.as_deref(), Some(""));
    assert_eq!(r.romaji, None);
}

#[test]
fn japanese_without_the_dictionary_names_the_catalog_id() {
    // The test process has no model store, so the dictionary is never installed here.
    let e = annotate(&ReadingsRequest {
        text: "東京".into(),
        lang: "ja".into(),
        hiragana: true,
        ..ReadingsRequest::default()
    })
    .unwrap_err();
    assert!(e.contains(japanese::DICTIONARY_ID), "{e}");
    let status = status_now();
    assert!(!status.japanese && status.chinese);
    assert_eq!(status.japanese_model.as_deref(), Some(japanese::DICTIONARY_ID));
}

#[test]
fn wire_format_is_camel_case() {
    let req: ReadingsRequest =
        serde_json::from_str(r#"{"text":"你好","lang":"zh","pinyin":true,"pinyinStyle":"numbers"}"#).unwrap();
    assert_eq!(req.pinyin_style, PinyinStyle::Numbers);
    assert!(!req.hiragana && !req.romaji && req.pinyin);
    let req: ReadingsRequest = serde_json::from_str(r#"{"text":"x","lang":"ja"}"#).unwrap();
    assert_eq!(req.pinyin_style, PinyinStyle::Marks);
    let result = ReadingsResult {
        lang: "ja".into(),
        hiragana: Some("とうきょう".into()),
        segments: vec![ReadingSegment { text: "東京".into(), reading: Some("とうきょう".into()) }],
        ..ReadingsResult::default()
    };
    assert_eq!(
        serde_json::to_value(&result).unwrap(),
        serde_json::json!({"lang":"ja","hiragana":"とうきょう","romaji":null,"pinyin":null,"segments":[{"text":"東京","reading":"とうきょう"}]})
    );
    let status = serde_json::to_value(ReadingsStatus {
        japanese_model: Some("x".into()),
        ..ReadingsStatus::default()
    })
    .unwrap();
    assert_eq!(status, serde_json::json!({"japanese":false,"chinese":false,"japaneseModel":"x","error":null}));
    assert_eq!(serde_json::to_value(PinyinStyle::None).unwrap(), "none");
}

#[test]
fn readings_are_clamped_to_the_event_limits() {
    let short = "かな".to_string();
    assert_eq!(clamp_reading(short.clone()), short);
    let clamped = clamp_reading("あ".repeat(MAX_READING_UNITS + 10));
    assert_eq!(clamped.encode_utf16().count(), MAX_READING_UNITS);
    assert!(clamped.ends_with('…'));
    // Astral characters count two UTF-16 units and are never split.
    let clamped = clamp_reading("𠀋".repeat(MAX_READING_UNITS));
    assert!(clamped.encode_utf16().count() <= MAX_READING_UNITS);
}

#[test]
fn segments_merge_and_cap() {
    let seg = |t: &str, r: Option<&str>| ReadingSegment {
        text: t.into(),
        reading: r.map(str::to_string),
    };
    let merged = finish_segments(vec![
        seg("は", None),
        seg("", Some("x")),
        seg("、", None),
        seg("東", Some("とう")),
        seg("京", Some("京")),
        seg("。", None),
    ]);
    assert_eq!(merged, vec![seg("は、", None), seg("東", Some("とう")), seg("京。", None)]);
    let many: Vec<ReadingSegment> = (0..MAX_SEGMENTS + 50).map(|i| seg(&format!("{i}"), Some("r"))).collect();
    let capped = finish_segments(many);
    assert_eq!(capped.len(), MAX_SEGMENTS);
    assert!(capped.last().unwrap().reading.is_none());
    let spelled: String = capped.iter().map(|s| s.text.as_str()).collect();
    let want: String = (0..MAX_SEGMENTS + 50).map(|i| i.to_string()).collect();
    assert_eq!(spelled, want);
}

// ---------------------------------------------------------------------------------------------
// Japanese: NJD features recorded from jpreprocess 0.15 + NAIST-JDIC (`dump_njd_features`)
// ---------------------------------------------------------------------------------------------

/// One Japanese fixture: the text, its NJD features, and the expected readings.
struct JaFixture {
    text: &'static str,
    features: &'static [&'static str],
    hiragana: &'static str,
    romaji: &'static str,
    /// Ruby segments as `text` or `text(reading)`, joined by `|`.
    segments: &'static str,
}

#[rustfmt::skip]
const JA_FIXTURES: &[JaFixture] = &[
    JaFixture {
        text: "私は学生です。",
        features: &[
            "私,名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ,0/3,C3,-1",
            "は,助詞,係助詞,*,*,*,*,は,ハ,ワ,0/1,動詞%F2/形容詞%F2/名詞%F1,1",
            "学生,名詞,一般,*,*,*,*,学生,ガクセイ,ガク’セー,5/4,C2,0",
            "です,助動詞,*,*,*,特殊・デス,基本形,です,デス,デス’,1/2,動詞%F1/形容詞%F2/名詞%F2@1,1",
            "。,記号,句点,*,*,*,*,。,、,、,0/0,*,0",
        ],
        hiragana: "わたしはがくせいです。",
        romaji: "watashi wa gakusei desu.",
        segments: "私(わたし)|は|学生(がくせい)|です。",
    },
    JaFixture {
        text: "東京に行きました。",
        features: &[
            "東京,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー,0/4,C2,-1",
            "に,助詞,格助詞,一般,*,*,*,に,ニ,ニ,0/1,動詞%F5/形容詞%F1/名詞%F1,1",
            "行き,動詞,自立,*,*,五段・カ行促音便,連用形,行き,イキ,イキ,3/2,*,0",
            "まし,助動詞,*,*,*,特殊・マス,連用形,まし,マシ,マシ’,1/2,動詞%F4@1/助詞%F2@1,1",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
            "。,記号,句点,*,*,*,*,。,、,、,0/0,*,0",
        ],
        hiragana: "とうきょうにいきました。",
        romaji: "tōkyō ni ikimashita.",
        segments: "東京(とうきょう)|に|行(い)|きました。",
    },
    JaFixture {
        text: "今日はいい天気ですね",
        features: &[
            "今日,名詞,副詞可能,*,*,*,*,今日,キョウ,キョー,1/2,C3,-1",
            "は,助詞,係助詞,*,*,*,*,は,ハ,ワ,0/1,動詞%F2/形容詞%F2/名詞%F1,1",
            "いい,動詞,自立,*,*,五段・ワ行促音便,連用形,いい,イイ,イイ,1/2,*,0",
            "天気,名詞,一般,*,*,*,*,天気,テンキ,テンキ,1/3,C1,0",
            "です,助動詞,*,*,*,特殊・デス,基本形,です,デス,デス’,1/2,動詞%F1/形容詞%F2/名詞%F2@1,1",
            "ね,助詞,終助詞,*,*,*,*,ね,ネ,ネ,1/1,動詞%F1/形容詞%F1/名詞%F1,1",
        ],
        hiragana: "きょうはいいてんきですね",
        romaji: "kyō wa ii tenki desu ne",
        segments: "今日(きょう)|はいい|天気(てんき)|ですね",
    },
    JaFixture {
        // A number with its counter: the digit is rewritten to 三, 本 is voiced (ぼん).
        text: "3本のペンを買った",
        features: &[
            "三,名詞,数,*,*,*,*,三,サン,サン,2/2,C3,0",
            "本,名詞,接尾,助数詞,*,*,*,本,ホン,ボン,1/2,C3,1",
            "の,助詞,連体化,*,*,*,*,の,ノ,ノ,0/1,動詞%F2/形容詞%F1,1",
            "ペン,名詞,一般,*,*,*,*,ペン,ペン,ペン,1/2,C3,0",
            "を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ,0/1,動詞%F5/名詞%F1,1",
            "買っ,動詞,自立,*,*,五段・ワ行促音便,連用タ接続,買っ,カッ,カッ,3/2,*,0",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
        ],
        hiragana: "さんぼんのぺんをかった",
        romaji: "sanbon no pen o katta",
        segments: "3(さん)|本(ぼん)|のペンを|買(か)|った",
    },
    JaFixture {
        // Latin words keep their own spelling everywhere; the auxiliary う lengthens 遊ぼ.
        text: "VRChatで遊ぼう！",
        features: &[
            "ＶＲＣｈａｔ,フィラー,*,*,*,*,*,ＶＲＣｈａｔ,ブイアールシーエイチエーティー,ブイアールシーエイチエーティー,0/14,*,-1",
            "で,助詞,格助詞,一般,*,*,*,で,デ,デ,1/1,動詞%F1,1",
            "遊ぼ,動詞,自立,*,*,五段・バ行,未然ウ接続,遊ぼ,アソボ,アソボ,0/3,*,0",
            "う,助動詞,*,*,*,不変化型,基本形,う,ウ,ー,0/1,動詞%F1,1",
            "！,記号,一般,*,*,*,*,！,、,、,0/0,*,0",
        ],
        hiragana: "VRChatであそぼう！",
        romaji: "VRChat de asobō!",
        segments: "VRChatで|遊(あそ)|ぼう！",
    },
    JaFixture {
        text: "食べている",
        features: &[
            "食べ,動詞,自立,*,*,一段,連用形,食べ,タベ,タベ,2/2,*,-1",
            "て,助詞,接続助詞,*,*,*,*,て,テ,テ,0/1,動詞%F1/形容詞%F1/名詞%F5,1",
            "いる,動詞,非自立,*,*,一段,基本形,いる,イル,イル,0/2,動詞%F4@1,0",
        ],
        hiragana: "たべている",
        romaji: "tabete iru",
        segments: "食(た)|べている",
    },
    JaFixture {
        text: "田中さんがコーヒーを飲む",
        features: &[
            "田中,名詞,固有名詞,人名,姓,*,*,田中,タナカ,タナカ,0/3,C1,-1",
            "さん,名詞,接尾,人名,*,*,*,さん,サン,サン,0/2,*,1",
            "が,助詞,格助詞,一般,*,*,*,が,ガ,ガ,0/1,名詞%F1,1",
            "コーヒー,名詞,一般,*,*,*,*,コーヒー,コーヒー,コーヒー,3/4,C1,0",
            "を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ,0/1,動詞%F5/名詞%F1,1",
            "飲む,動詞,自立,*,*,五段・マ行,基本形,飲む,ノム,ノム,1/2,*,0",
        ],
        hiragana: "たなかさんがこーひーをのむ",
        romaji: "tanaka-san ga kōhī o nomu",
        segments: "田中(たなか)|さんがコーヒーを|飲(の)|む",
    },
    JaFixture {
        // 1人 becomes one node 一人 (ひとり); 20 becomes 二 + 十 with the sound change of 歳.
        text: "1人で20歳になった",
        features: &[
            "一人,名詞,副詞可能,*,*,*,*,一人,ヒトリ,ヒ’トリ,2/3,*,-1",
            "で,助詞,格助詞,一般,*,*,*,で,デ,デ,1/1,動詞%F1,1",
            "二,名詞,数,*,*,*,*,二,ニ,ニ,3/1,C3,0",
            "十,名詞,数,*,*,*,*,十,ジュウ,ジュッ,1/2,*,1",
            "歳,名詞,接尾,助数詞,*,*,*,歳,サイ,サイ,1/2,C3,1",
            "に,助詞,格助詞,一般,*,*,*,に,ニ,ニ,0/1,動詞%F5/形容詞%F1/名詞%F1,1",
            "なっ,動詞,自立,*,*,五段・ラ行,連用タ接続,なっ,ナッ,ナッ,1/2,*,0",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
        ],
        hiragana: "ひとりでにじゅっさいになった",
        romaji: "hitori de nijussai ni natta",
        segments: "1人(ひとり)|で|20(にじゅっ)|歳(さい)|になった",
    },
    JaFixture {
        text: "お茶を飲みたい",
        features: &[
            "お茶,名詞,一般,*,*,*,*,お茶,オチャ,オチャ,0/2,C3,-1",
            "を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ,0/1,動詞%F5/名詞%F1,1",
            "飲み,動詞,自立,*,*,五段・マ行,連用形,飲み,ノミ,ノミ,3/2,*,0",
            "たい,助動詞,*,*,*,特殊・タイ,基本形,たい,タイ,タイ,1/2,動詞%F4@1,1",
        ],
        hiragana: "おちゃをのみたい",
        romaji: "ocha o nomitai",
        segments: "お|茶(ちゃ)|を|飲(の)|みたい",
    },
    JaFixture {
        text: "大きい家",
        features: &[
            "大きい,形容詞,自立,*,*,形容詞・イ段,基本形,大きい,オオキイ,オーキイ,3/4,*,-1",
            "家,名詞,一般,*,*,*,*,家,イエ,イエ,2/2,C3,0",
        ],
        hiragana: "おおきいいえ",
        romaji: "ōkii ie",
        segments: "大(おお)|きい|家(いえ)",
    },
    JaFixture {
        text: "「はい」と言った",
        features: &[
            "「,記号,括弧開,*,*,*,*,「,、,、,0/0,*,-1",
            "はい,感動詞,*,*,*,*,*,はい,ハイ,ハイ,1/2,C3,0",
            "」,記号,括弧閉,*,*,*,*,」,、,、,0/0,*,0",
            "と,助詞,格助詞,引用,*,*,*,と,ト,ト,0/1,動詞%F1/形容詞%F1,0",
            "言っ,動詞,自立,*,*,五段・ワ行促音便,連用タ接続,言っ,イッ,イッ,3/2,*,0",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
        ],
        hiragana: "「はい」といった",
        romaji: "\"hai\" to itta",
        segments: "「はい」と|言(い)|った",
    },
    JaFixture {
        // The comma inside the number is dropped by the number rules; 円 joins 千 with n'.
        text: "1,000円です",
        features: &[
            "千,名詞,数,*,*,*,*,千,セン,セン,2/2,*,0",
            "円,名詞,接尾,助数詞,*,*,*,円,エン,エン,1/2,C3,1",
            "です,助動詞,*,*,*,特殊・デス,基本形,です,デス,デス’,1/2,動詞%F1/形容詞%F2/名詞%F2@1,1",
        ],
        hiragana: "せんえんです",
        romaji: "sen'en desu",
        segments: "1,000(せん)|円(えん)|です",
    },
    JaFixture {
        // Half-width katakana with separate voicing marks: the segments keep the original text.
        text: "ｶﾞｷﾞｸﾞ",
        features: &[
            "ガ,名詞,一般,*,*,*,*,ガ,ガ,ガ,0/1,C3,-1",
            "ギグ,名詞,一般,*,*,*,*,ギグ,ギグ,ギグ,1/2,C5,1",
        ],
        hiragana: "がぎぐ",
        romaji: "ga gigu",
        segments: "ｶﾞｷﾞｸﾞ",
    },
    JaFixture {
        // A year: digits become 二千二十四 (四年 is よねん); 1月 and 1日 (ついたち) are dictionary words.
        text: "2024年1月1日",
        features: &[
            "二,名詞,数,*,*,*,*,二,ニ,ニ,2/1,C3,0",
            "千,名詞,数,*,*,*,*,千,セン,セン,1/2,*,1",
            "二,名詞,数,*,*,*,*,二,ニ,ニ,1/1,C3,0",
            "十,名詞,数,*,*,*,*,十,ジュウ,ジュー,1/2,*,1",
            "四,名詞,数,*,*,*,*,四,ヨン,ヨ,1/1,C1,0",
            "年,名詞,接尾,助数詞,*,*,*,年,ネン,ネン,1/2,C3,1",
            "１月,名詞,副詞可能,*,*,*,*,１月,イチガツ,イチガツ,4/4,C2,0",
            "一日,名詞,副詞可能,*,*,*,*,一日,ツイタチ,ツイタチ,4/4,*,0",
        ],
        hiragana: "にせんにじゅうよねんいちがつついたち",
        romaji: "nisennijūyonen ichigatsu tsuitachi",
        segments: "2024(にせんにじゅうよ)|年(ねん)|1月(いちがつ)|1日(ついたち)",
    },
    JaFixture {
        text: "ちょっと待って",
        features: &[
            "ちょっと,副詞,助詞類接続,*,*,*,*,ちょっと,チョット,チョット,1/3,*,-1",
            "待っ,動詞,自立,*,*,五段・タ行,連用タ接続,待っ,マッ,マッ,1/2,*,0",
            "て,助詞,接続助詞,*,*,*,*,て,テ,テ,0/1,動詞%F1/形容詞%F1/名詞%F5,1",
        ],
        hiragana: "ちょっとまって",
        romaji: "chotto matte",
        segments: "ちょっと|待(ま)|って",
    },
    JaFixture {
        // A dictionary word mixing a letter with kanji: the letter as written, the kanji read.
        text: "Ｘ線を撮った",
        features: &[
            "Ｘ線,名詞,一般,*,*,*,*,Ｘ線,エックスセン,エック’スセン,0/6,C2,-1",
            "を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ,0/1,動詞%F5/名詞%F1,1",
            "撮っ,動詞,自立,*,*,五段・ラ行,連用タ接続,撮っ,トッ,トッ,1/2,*,0",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
        ],
        hiragana: "Ｘせんをとった",
        romaji: "X-sen o totta",
        segments: "Ｘ|線(せん)|を|撮(と)|った",
    },
    JaFixture {
        // A letter with katakana (typed half-width): no kanji, so no ruby; the kana are read.
        text: "Tシャツを買った",
        features: &[
            "Ｔシャツ,名詞,一般,*,*,*,*,Ｔシャツ,ティーシャツ,ティーシャツ,0/4,C2,-1",
            "を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ,0/1,動詞%F5/名詞%F1,1",
            "買っ,動詞,自立,*,*,五段・ワ行促音便,連用タ接続,買っ,カッ,カッ,3/2,*,0",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
        ],
        hiragana: "Tしゃつをかった",
        romaji: "T-shatsu o katta",
        segments: "Tシャツを|買(か)|った",
    },
    JaFixture {
        text: "Jリーグが好き",
        features: &[
            "Ｊリーグ,名詞,固有名詞,組織,*,*,*,Ｊリーグ,ジェイリーグ,ジェイリーグ,3/5,C1,-1",
            "が,助詞,格助詞,一般,*,*,*,が,ガ,ガ,0/1,名詞%F1,1",
            "好き,名詞,形容動詞語幹,*,*,*,*,好き,スキ,ス’キ,2/2,C4,0",
        ],
        hiragana: "Jりーぐがすき",
        romaji: "J-rīgu ga suki",
        segments: "Jリーグが|好(す)|き",
    },
    JaFixture {
        // The long vowel of the kanji comes from its pronunciation (キュウ / キュー → kyū).
        text: "Ｂ級グルメ",
        features: &[
            "Ｂ級,名詞,一般,*,*,*,*,Ｂ級,ビーキュウ,ビーキュー,5/4,*,-1",
            "グルメ,名詞,一般,*,*,*,*,グルメ,グルメ,グルメ,1/3,C1,1",
        ],
        hiragana: "Ｂきゅうぐるめ",
        romaji: "B-kyū gurume",
        segments: "Ｂ|級(きゅう)|グルメ",
    },
    JaFixture {
        // The letter is not read by its name (ワールドカップ): the word is read whole.
        text: "Ｗ杯",
        features: &["Ｗ杯,名詞,一般,*,*,*,*,Ｗ杯,ワールドカップ,ワールドカップ,5/7,*,-1"],
        hiragana: "わーるどかっぷ",
        romaji: "wārudokappu",
        segments: "Ｗ杯(わーるどかっぷ)",
    },
    JaFixture {
        // An all-Latin word keeps its spelling; the romaji line narrows full-width letters.
        text: "ＣＤを買った",
        features: &[
            "ＣＤ,名詞,一般,*,*,*,*,ＣＤ,シーディー,シーディー,3/4,C1,-1",
            "を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ,0/1,動詞%F5/名詞%F1,1",
            "買っ,動詞,自立,*,*,五段・ワ行促音便,連用タ接続,買っ,カッ,カッ,3/2,*,0",
            "た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ,0/1,動詞%F2@1/形容詞%F4@-2,1",
        ],
        hiragana: "ＣＤをかった",
        romaji: "CD o katta",
        segments: "ＣＤを|買(か)|った",
    },
];

/// Segments as `text` / `text(reading)` joined by `|`.
fn segments_string(segments: &[ReadingSegment]) -> String {
    segments
        .iter()
        .map(|s| match &s.reading {
            Some(r) => format!("{}({r})", s.text),
            None => s.text.clone(),
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn check_japanese(text: &str, morphs: &[japanese::Morph], fx: &JaFixture, failures: &mut Vec<String>) {
    let r = japanese::read(text, morphs, true, true);
    let segments = finish_segments(r.segments.clone());
    let spelled: String = segments.iter().map(|s| s.text.as_str()).collect();
    for (what, got, want) in [
        ("hiragana", r.hiragana.clone().unwrap_or_default(), fx.hiragana.to_string()),
        ("romaji", r.romaji.clone().unwrap_or_default(), fx.romaji.to_string()),
        ("segments", segments_string(&segments), fx.segments.to_string()),
        ("spelling", spelled, text.to_string()),
    ] {
        if got != want {
            failures.push(format!("{text} {what}: got {got:?}, want {want:?}"));
        }
    }
}

#[test]
fn japanese_fixtures_from_recorded_features() {
    let mut failures = Vec::new();
    for fx in JA_FIXTURES {
        let features: Vec<String> = fx.features.iter().map(|s| s.to_string()).collect();
        let morphs = japanese::Morph::parse_all(&features);
        assert_eq!(morphs.len(), fx.features.len(), "{}: every feature line parses", fx.text);
        check_japanese(fx.text, &morphs, fx, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn japanese_forms_follow_the_request() {
    let fx = &JA_FIXTURES[0];
    let features: Vec<String> = fx.features.iter().map(|s| s.to_string()).collect();
    let morphs = japanese::Morph::parse_all(&features);
    // Romaji only: romaji ruby.
    let r = japanese::read(fx.text, &morphs, false, true);
    assert_eq!(r.hiragana, None);
    assert_eq!(r.romaji.as_deref(), Some("watashi wa gakusei desu."));
    assert_eq!(segments_string(&finish_segments(r.segments)), "私(watashi)|は|学生(gakusei)|です。");
    // Hiragana only.
    let r = japanese::read(fx.text, &morphs, true, false);
    assert_eq!((r.hiragana.as_deref(), r.romaji.as_deref()), (Some("わたしはがくせいです。"), None));
}

#[test]
fn japanese_alignment_survives_unknown_text_and_missing_morphs() {
    // Text the analyzer dropped (spaces, emoji) stays in place; morphs past the text are ignored.
    let features = [
        "東京,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー,0/4,C2,-1".to_string(),
        "大阪,名詞,固有名詞,地域,一般,*,*,大阪,オオサカ,オーサカ,0/4,C2,-1".to_string(),
    ];
    let morphs = japanese::Morph::parse_all(&features);
    let r = japanese::read("東京 🙂 大阪", &morphs, true, true);
    assert_eq!(r.hiragana.as_deref(), Some("とうきょう 🙂 おおさか"));
    assert_eq!(r.romaji.as_deref(), Some("tōkyō 🙂 ōsaka"));
    let spelled: String = r.segments.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(spelled, "東京 🙂 大阪");
    // No morphs at all: everything is kept as written.
    let r = japanese::read("ABC", &[], true, true);
    assert_eq!((r.hiragana.as_deref(), r.romaji.as_deref()), (Some("ABC"), Some("ABC")));
    // Malformed feature lines are skipped.
    assert!(japanese::Morph::parse("not,a,feature").is_none());
}

/// Real dictionary: the analyzer gives the recorded features, and the readings match.
#[test]
#[ignore = "needs SIGIL_TEST_MODELS_DIR with naist-jdic (jpreprocess v0.15.0)"]
fn real_dictionary_reads_the_fixtures() {
    let Some(dir) = test_dictionary() else { return };
    let started = std::time::Instant::now();
    let analyzer = japanese::JapaneseAnalyzer::load(&dir).expect("load NAIST-JDIC");
    eprintln!("dictionary loaded in {} ms", started.elapsed().as_millis());
    let mut failures = Vec::new();
    for fx in JA_FIXTURES {
        let features = analyzer.features(fx.text).unwrap();
        let recorded: Vec<String> = fx.features.iter().map(|s| s.to_string()).collect();
        if features != recorded {
            failures.push(format!("{}: features changed:\n{}", fx.text, features.join("\n")));
        }
        check_japanese(fx.text, &analyzer.morphs(fx.text).unwrap(), fx, &mut failures);
    }
    // A long line stays fast.
    let long = "今日は東京で友達と美味しいラーメンを食べました。".repeat(40);
    let started = std::time::Instant::now();
    let r = japanese::read(&long, &analyzer.morphs(&long).unwrap(), true, true);
    eprintln!("{} chars read in {} ms", long.chars().count(), started.elapsed().as_millis());
    assert!(r.hiragana.is_some_and(|h| h.starts_with("きょうはとうきょうでともだちとおいしいらーめんをたべました。")));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn kana_words_without_a_dictionary_reading_read_as_spelled() {
    let features = ["ぴえん,名詞,一般,*,*,*,*,ぴえん,*,*,0/0,*,0".to_string()];
    let morphs = japanese::Morph::parse_all(&features);
    assert_eq!(morphs[0].read, None);
    let r = japanese::read("ぴえん", &morphs, true, true);
    assert_eq!((r.hiragana.as_deref(), r.romaji.as_deref()), (Some("ぴえん"), Some("pien")));
}

#[test]
fn a_broken_dictionary_is_an_error_not_a_panic() {
    let dir = std::env::temp_dir().join(format!("sigil-readings-broken-dict-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["char_def.bin", "dict.da", "dict.vals", "dict.words", "dict.wordsidx", "matrix.mtx", "metadata.json", "unk.bin"] {
        std::fs::write(dir.join(f), b"").unwrap();
    }
    assert_eq!(japanese::dictionary_dir(&dir), Some(dir.clone()));
    let loaded = std::panic::catch_unwind(|| japanese::JapaneseAnalyzer::load(&dir).map(|_| ()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(matches!(loaded, Ok(Err(_))), "a broken dictionary must fail to load cleanly");
    // A folder without the files is not a dictionary.
    assert_eq!(japanese::dictionary_dir(&std::env::temp_dir().join("sigil-no-such-dictionary")), None);
}
