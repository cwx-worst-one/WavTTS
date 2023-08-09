class MetadataMulanTextTransform():
    @staticmethod
    def _mcc_metadata_to_string(item):
        metadata = item['metadata']
        keys = ['final_genre', 'final_mood', 'final_theme']
        fields = []
        for k in keys:
            val = metadata.get(k)
            if val is None or val == 'nan': continue
            fields.extend(val.split(','))
        return ' '.join(fields)

    def __call__(self, item):
        metadata_string = MetadataMulanTextTransform._mcc_metadata_to_string(item)
        return {
            **item, 'text': metadata_string
        }