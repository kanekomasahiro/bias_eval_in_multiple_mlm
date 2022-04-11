# Gender Bias in Masked Language Models for Multiple Languages

Code and data for the paper: "Gender Bias in Masked Language Models for Multiple Languages" (In NAACL 2022). If you use any part of this work, make sure you include the following citation:
```
@inproceedings{Kaneko:NAACL2022,
    title = "Gender Bias in Masked Language Models for Multiple Languages",
    author = "Kaneko, Masahiro  and
      Imankulova, Aizhan  and
      Bollegala, Danushka  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)",
    month = July,
    year = "2022",
    address = "Seattle",
    publisher = "Association for Computational Linguistics",
}
```

## 🛠 Setup

All requirements can be found in `requirements.txt`. You can install all required packages with `pip install -r requirements.txt`.

## 🖥 Evaluating MLMs in multiple languages

You can evaluate the bias using --corpus to select the a parallel corpus and --lang to select the languages.

```
python eval.py --corpus [ted, news] --lang [de, ja, ar, es, pt, ru, id, zh] --method aula 
```

## 💻 Japanese and Russian corpora to evaluate social biases 

`japanese.json` and `russian.json` are manually translated data from Crows-Pairs into Japanese and Russian, respectively. You can use [this code](https://github.com/kanekomasahiro/evaluate_bias_in_mlm) to evaluate bias for them.

## 📜 License

See the LICENSE file
