
entries = [
    ('str', 'model', 'model path'),

]

tmpl = '''
    @property
    def {name}(self) -> {type}:
        """{doc}"""
        return self.p.{name}.decode()

    @{name}.setter
    def {name}(self, value: {type}):
        self.p.{name} = value.encode('utf8')
'''

for type, name, doc in entries:
    print(tmpl.format(name=name, type=type, doc=doc))






