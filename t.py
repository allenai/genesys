import model_discovery.utils as U

dir='/home/junyanc/model_discovery/ckpt/evolution_test/db/DLAM3/artifact.json'

a=U.load_json(dir)


code=a['code'].split('\n')
for idx, line in enumerate(code):
    if 'class GAB(nn.Module):' in line:
        break
code[idx]='from .block_registry import BlockRegister\n\n__all__ = [\n    "GAB",\n]\n@BlockRegister(\n    name="default",\n    config={}\n)class GAB(nn.Module):'
code='\n'.join(code)
print(code)

