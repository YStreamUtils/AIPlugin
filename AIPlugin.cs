using Microsoft.Extensions.AI;
using OllamaSharp;
using YStreamUtils.SDK.Plugin;

[assembly: PluginManifest(
    "AI-Plugin", 
    "v1.0.6", 
    "The official YStreamUtils AI Chat Integration",
    "YStreamUtils",
    "AIPlugin",
    ["network", "AI"],
    ["ysnt-yes"],
    typeof(AIPlugin.AiPluginSettings))]

namespace AIPlugin;

/// <summary>
/// Configuration settings for the <see cref="AiPlugin"/>.
/// </summary>
public class AiPluginSettings
{
    /// <summary>
    /// Gets or sets the base URI of the Ollama API server.
    /// </summary>
    public string Endpoint { get; set; } = "http://localhost:11434/";

    /// <summary>
    /// Gets or sets the identifier of the Ollama model to use for inference (e.g., <c>llama3.2</c>).
    /// </summary>
    public string ModelId { get; set; } = "llama3.2";

    /// <summary>
    /// Gets or sets the default system instructions applied to all prompts unless overridden per request.
    /// </summary>
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";

    /// <summary>
    /// Gets or sets the maximum number of <see cref="ChatMessage"/> entries retained in a conversation's history before older messages are trimmed.
    /// </summary>
    public int MaxHistoryMessages { get; set; } = 30;

    /// <summary>
    /// Gets or sets the sliding expiration duration for persisted conversation histories in the cache.
    /// </summary>
    public TimeSpan ConversationTtl { get; set; } = TimeSpan.FromHours(6);
}

/// <summary>
/// Provides AI chat capabilities backed by Ollama, supporting both stateless one-off prompts and persistent multi-turn conversations.
/// </summary>
/// <param name="cache">The isolated cache provider scoped to <see cref="AiPlugin"/> for persisting conversation histories.</param>
public class AiPlugin(ICacheProvider<AiPlugin> cache) : PluginBase<AiPluginSettings>
{
    private IChatClient? _client;
    private ChatOptions _defaultOptions = new();

    /// <summary>
    /// Initializes the plugin and constructs the underlying <see cref="IChatClient"/> using the current <see cref="PluginBase{TSettings}.Settings"/>.
    /// </summary>
    /// <returns>A completed <see cref="Task"/> once the client is ready.</returns>
    public override Task InitializeAsync()
    {
        BuildClient(Settings);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Rebuilds the <see cref="IChatClient"/> and default chat options whenever plugin settings are updated at runtime.
    /// </summary>
    /// <param name="settings">The updated <see cref="AiPluginSettings"/> instance.</param>
    /// <returns>A completed <see cref="Task"/> once the client has been reconfigured.</returns>
    protected override Task OnSettingsUpdatedAsync(AiPluginSettings settings)
    {
        BuildClient(settings);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Disposes any existing <see cref="IChatClient"/> and instantiates a new <see cref="OllamaApiClient"/> from the provided settings.
    /// </summary>
    /// <param name="settings">The settings containing the endpoint, model ID, and default system prompt.</param>
    private void BuildClient(AiPluginSettings settings)
    {
        _client?.Dispose();

        _client = new OllamaApiClient(new Uri(settings.Endpoint), settings.ModelId);
        _defaultOptions = new ChatOptions
        {
            Instructions = settings.SystemPrompt
        };
    }

    /// <summary>
    /// Sends a stateless, one-off prompt to the AI model without reading from or writing to the conversation cache.
    /// </summary>
    /// <param name="prompt">The user message or question to send to the model.</param>
    /// <param name="systemPrompt">Optional system instructions to override <see cref="AiPluginSettings.SystemPrompt"/> for this request only.</param>
    /// <param name="ct">A <see cref="CancellationToken"/> to cancel the asynchronous request.</param>
    /// <returns>The assistant's plain-text response.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the plugin has not been initialized prior to calling this method.</exception>
    public async Task<string> AskAsync(
        string prompt, 
        string? systemPrompt = null, 
        CancellationToken ct = default)
    {
        if (_client is null)
            throw new InvalidOperationException("Plugin not initialized.");

        var options = systemPrompt is not null 
            ? new ChatOptions { Instructions = systemPrompt } 
            : _defaultOptions;

        var response = await _client.GetResponseAsync(prompt, options, cancellationToken: ct);
        return response.Text;
    }

    /// <summary>
    /// Sends a user message within a stateful, multi-turn conversation, persisting the message history in the cache under <paramref name="conversationId"/>.
    /// </summary>
    /// <remarks>
    /// Prior messages are loaded from <see cref="ICacheProvider{T}"/>, trimmed to <see cref="AiPluginSettings.MaxHistoryMessages"/> if necessary,
    /// and updated with both the user's message and all assistant/tool messages returned by the model.
    /// </remarks>
    /// <param name="conversationId">A unique identifier representing the conversation thread (e.g., a channel ID or user session ID).</param>
    /// <param name="userMessage">The new message from the user to append to the conversation.</param>
    /// <param name="systemPrompt">Optional system instructions to override <see cref="AiPluginSettings.SystemPrompt"/> for this turn.</param>
    /// <param name="ct">A <see cref="CancellationToken"/> to cancel the asynchronous request.</param>
    /// <returns>The assistant's plain-text response for the current turn.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the plugin has not been initialized prior to calling this method.</exception>
    public async Task<string> ChatAsync(
        string conversationId, 
        string userMessage, 
        string? systemPrompt = null, 
        CancellationToken ct = default)
    {
        if (_client is null)
            throw new InvalidOperationException("Plugin not initialized.");

        var history = await cache.GetAsync<List<ChatMessage>>(conversationId, ct) ?? [];
        history.Add(new ChatMessage(ChatRole.User, userMessage));

        if (history.Count > Settings.MaxHistoryMessages)
            history = history.TakeLast(Settings.MaxHistoryMessages).ToList();

        var options = systemPrompt is not null 
            ? new ChatOptions { Instructions = systemPrompt } 
            : _defaultOptions;

        var response = await _client.GetResponseAsync(history, options, cancellationToken: ct);
        history.AddRange(response.Messages);

        await cache.SetAsync(conversationId, history, Settings.ConversationTtl, ct);
        return response.Text;
    }

    /// <summary>
    /// Retrieves the cached message history for a specific conversation.
    /// </summary>
    /// <param name="conversationId">The unique identifier of the conversation to retrieve.</param>
    /// <param name="ct">A <see cref="CancellationToken"/> to cancel the asynchronous cache lookup.</param>
    /// <returns>A read-only list of <see cref="ChatMessage"/> entries in chronological order, or an empty list if no history exists.</returns>
    public async Task<IReadOnlyList<ChatMessage>> GetHistoryAsync(string conversationId, CancellationToken ct = default)
        => await cache.GetAsync<List<ChatMessage>>(conversationId, ct) ?? [];

    /// <summary>
    /// Deletes the stored message history for a specific conversation from the cache.
    /// </summary>
    /// <param name="conversationId">The unique identifier of the conversation to clear.</param>
    /// <param name="ct">A <see cref="CancellationToken"/> to cancel the asynchronous cache removal.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous removal operation.</returns>
    public Task ClearConversationAsync(string conversationId, CancellationToken ct = default)
        => cache.RemoveAsync(conversationId, ct);

    /// <summary>
    /// Releases the underlying <see cref="IChatClient"/> resources when the plugin is unloaded or shut down.
    /// </summary>
    /// <returns>A completed <see cref="ValueTask"/> once disposal is finished.</returns>
    protected override ValueTask OnDisposeAsync()
    {
        _client?.Dispose();
        _client = null;
        return ValueTask.CompletedTask;
    }
}
