use anyhow::Result;
use rig::{
    Embed,
    client::{CompletionClient, EmbeddingsClient, ProviderClient},
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai,
};
use rig_surrealdb::{Mem, SurrealVectorStore};
use serde::Serialize;
use surrealdb::Surreal;

#[derive(Embed, Serialize)]
struct WordDefinition {
    word: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let surreal = Surreal::new::<Mem>(()).await?;
    surreal.use_ns("ns").use_db("db").await?;
    let client = openai::Client::from_env();
    let model = client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);

    let vector_store = SurrealVectorStore::with_defaults(model.clone(), surreal.clone());

    let words = vec![
        WordDefinition {
            word: "flurbo".to_string(),
            definition: "A fictional currency from Rick and Morty.".to_string(),
        },
        WordDefinition {
            word: "glarb-glarb".to_string(),
            definition: "A creature from the marshlands of Glibbo.".to_string(),
        },
        WordDefinition {
            word: "wubba-lubba".to_string(),
            definition: "A catchphrase popularized by Rick Sanchez.".to_string(),
        },
        WordDefinition {
            word: "schmeckle".to_string(),
            definition: "A small unit of currency in some fictional universes.".to_string(),
        },
        WordDefinition {
            word: "plumbus".to_string(),
            definition: "A common household device with an unclear purpose.".to_string(),
        },
        WordDefinition {
            word: "zorp".to_string(),
            definition: "A term used to describe an alien greeting.".to_string(),
        },
        WordDefinition {
            word: "gorg".to_string(),
            definition:
                "A family of giants in the series Fraggle Rock that live in a rundown castle."
                    .to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model)
        .documents(words)
        .unwrap()
        .build()
        .await?;

    vector_store.insert_documents(documents).await?;

    let linguist_agent = client
        .agent(openai::GPT_4_1_NANO)
        .preamble("You are a linguist. If you don't know don't make up an answer.")
        .dynamic_context(3, vector_store)
        .build();

    let prompts = vec![
        "What is a zorp?",
        "What's the word that corresponds to a small unit of currency?",
        "What is a gloubi-boulga?",
        "What sort of castle do giants live in?",
    ];

    for prompt in prompts {
        let response = linguist_agent.prompt(prompt).await?;
        println!("{}", response);
    }

    Ok(())
}
