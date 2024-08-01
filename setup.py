from setuptools import setup, find_packages


def get_requirements(fns, envsub: bool = False):
    reg = r'\$(\w+)'
    reqs = []
    for fn in fns:
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Given file {fn} does not exists.')
        with open(fn, 'r') as f:
            for line in f.readlines():
                s = line.strip()
                if s.startswith('#'):
                    continue
                if envsub:
                    for k in re.findall(reg, line):
                        v = os.environ.get(k)
                        if v is None:
                            warnings.warn(
                                f'Environment variable "{k}" is required by "{s}"'
                                f'but not given. Skip'
                            )
                            break
                        s = s.replace('$'+k, v)
                    else:
                        reqs.append(s)


setup(
    name="llm_retrieval_qa",
    version="0.0.1",
    packages=find_packages(exclude=["examples"]),
    install_requires=get_requirements(["requirements.txt"]),
)
