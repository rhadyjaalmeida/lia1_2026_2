"""
Funções de treinamento de modelos para o projeto RetentionAI.

Este módulo é um ponto de extensão para quando o pipeline de treinamento
(pré-processamento + modelo, via sklearn.pipeline.Pipeline) estiver
validado no notebook e pronto para ser reutilizado/versionado.

O objetivo final é permitir salvar o pipeline completo (pré-processamento
+ estimador) em `models/`, de modo que a inferência em novos dados possa
ser feita sem repetir manualmente o pré-processamento:

    import joblib
    pipeline = joblib.load("models/pipeline_final.joblib")
    pipeline.predict_proba(novos_clientes)
"""
