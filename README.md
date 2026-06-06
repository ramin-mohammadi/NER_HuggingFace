# Name Entity Recognition (NER) on FiNER Dataset
### Dataset
 - A sequence is a single sentence
 - Every word is given an NER label 
 - NER tags: Person, Organization, Location, non NER tag
- 6 labels
  - {'O': 0, 'PER_B': 1, 'PER_I': 2, 'LOC_B': 3, 'LOC_I': 4, 'ORG_B': 5, 'ORG_I': 6}
  - O is not an NER tag
  - Multi token entities are split into separate tokens
   - Ex: "United States of America is a country." -> [LOC_B] [LOC_I] [LOC_I] [LOC_I] [O] [O] [O].
   - so every NER tag has a ..._B (beginning) and ..._I (intermediate)
 ### Results
- NER seen as a classification task
   - NLL Loss or CE loss
- Underlying problem is determining which tokens in a multi token entity is the beginning and is the last intermediate (requires understanding of  context and long spatial awareness) -> BERT appropriate (Transformer and allows fine tuning for classifcation task like NER)
- Fine Tuned Hugging Face BERT-base model in 5 epochs and achieved 0.96 F1 score on test set
- Documentation and understanding can be found <a href="./Understanding_NER_HuggingFace.pdf">here</a> 
