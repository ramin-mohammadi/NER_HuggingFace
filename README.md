# Name Entity Recognition (NER) on FiNER Dataset
- Dataset
 - A sequence is a single sentence
 - Every word is given an NER label
 - Multi token entities are split into separate tokens
   - Ex: "United States is a country." -> [LOC] [LOC] [0] [0] [0]. 
 - NER tags: Person, Organization, Location, non NER tag
- 6 labels
  - {'O': 0, 'PER_B': 1, 'PER_I': 2, 'LOC_B': 3, 'LOC_I': 4, 'ORG_B': 5, 'ORG_I': 6}
- NER seen as a classification task
- Fine Tuned Hugging Face BERT-base model in 5 epochs and achieved 0.96 F1 score on test set

  - 


