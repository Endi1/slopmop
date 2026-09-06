use std::path::Path;

use anyhow::{Context, Result};
use tree_sitter::{Node, Parser};

use super::CodeEntity;

pub(super) fn supports(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "go")
}

pub(super) fn parser() -> Result<Parser> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .context("failed to load Go grammar")?;
    Ok(parser)
}

pub(super) fn parse(parser: &mut Parser, source: &str) -> Result<Vec<CodeEntity>> {
    let tree = parser
        .parse(source, None)
        .context("Go parser failed to produce a syntax tree")?;
    let mut entities = Vec::new();
    collect_entities(tree.root_node(), source, &mut entities);
    Ok(entities)
}

fn collect_entities(node: Node<'_>, source: &str, entities: &mut Vec<CodeEntity>) {
    let is_indexed_node = match node.kind() {
        "function_declaration" | "method_declaration" => true,
        "type_spec" => node
            .child_by_field_name("type")
            .is_some_and(|node| matches!(node.kind(), "struct_type" | "interface_type")),
        _ => false,
    };

    if is_indexed_node {
        let name = node
            .child_by_field_name("name")
            .and_then(|name| name.utf8_text(source.as_bytes()).ok())
            .unwrap_or("<unknown>");
        let content = node
            .utf8_text(source.as_bytes())
            .expect("Go code entity content was not valid UTF-8");

        entities.push(CodeEntity {
            name: name.to_owned(),
            content: content.to_owned(),
        });
        return;
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_entities(child, source, entities);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_supported_go_entities() -> Result<()> {
        let source = r#"
package example

type Item struct{}
type Reader interface { Read() }
type Alias string
func Run() {}
func (Item) Method() {}
"#;

        let entities = parse(&mut parser()?, source)?;
        let names = entities
            .iter()
            .map(|entity| entity.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, ["Item", "Reader", "Run", "Method"]);
        Ok(())
    }
}
