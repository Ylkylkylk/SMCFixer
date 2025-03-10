# SMCFixer

## Overall Framework
SMCFIXER comprising three key modules: code slicing, knowledge retrieval, and patch generation. Given a Solidity code file, we first use Remix [18] to attempt compilation. If the file fails to compile, our approach automatically generates patches by leveraging the compiler’s error messages and relevant expert knowledge. 
(./picture/framework.png)
