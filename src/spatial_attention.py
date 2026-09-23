import ast
import inspect
import textwrap
import types

def local_patches(value, size, margin, n):
    (batch, channels, height, width) = value.shape
    assert height == width == n * size
    patches = value.reshape(batch, channels, n, size, n, size).permute(0, 2, 4, 1, 3, 5)
    patches = patches[:, margin[0]:n - margin[1], margin[2]:n - margin[3]]
    return patches.reshape(batch, -1, channels, size, size)

def install_vectorized_patches(model):
    module = model.backbone.att
    original = module.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    matches = [node for node in tree.body[0].body if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and (node.targets[0].id == 'p') and isinstance(node.value, ast.Call) and (ast.unparse(node.value.func) == 'torch.stack')]
    assert len(matches) == 1, 'Archived GL patch extraction changed'
    assignment = matches[0]
    assert isinstance(assignment.value.args[0], ast.ListComp)
    replacement = ast.parse('_local_patches(L,self.size,self.margin,self.N)', mode='eval').body
    assignment.value = ast.IfExp(test=ast.parse('self.vectorized_patches', mode='eval').body, body=replacement, orelse=assignment.value)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__func__.__globals__, _local_patches=local_patches)
    exec(compile(tree, '<vectorized_gl_patch_extraction>', 'exec'), namespace)
    module.forward = types.MethodType(namespace[original.__name__], module)
    module.vectorized_patches = False
    return ast.unparse(tree)
