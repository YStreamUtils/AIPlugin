using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using OllamaSharp;
using YStreamUtils.SDK.Plugin;


[assembly: PluginManifest(
    "AI-Plugin", 
    "v1.0.2", 
    "The official YStreamUtils AI Chat Integration",
    "YStreamUtils",
    "AIPlugin",
    ["network", "AI"],
    ["ysnt-yes"],
    typeof(AIPlugin.AiPluginSettings))]

namespace AIPlugin;
public class AiPluginSettings
{
    public string Endpoint { get; set; } = "http://localhost:11434/";
    public string ModelId { get; set; } = "llama3.2";
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";
}


public class AiPlugin : PluginBase<AiPluginSettings>
{
    private IChatClient? _client;
    private ChatOptions _options = new();
    
    public AiPlugin() { }

    public override Task InitializeAsync()
    {
        BuildClient(Settings);
        return Task.CompletedTask;
    }

    protected override Task OnSettingsUpdatedAsync(AiPluginSettings settings)
    {
        BuildClient(settings);
        return Task.CompletedTask;
    }

    private void BuildClient(AiPluginSettings settings)
    {
        _client?.Dispose();

        _client = new OllamaApiClient(new Uri(settings.Endpoint), settings.ModelId);
        _options = new ChatOptions
        {
            Instructions = settings.SystemPrompt
        };
    }
    
    public async Task<string> AskAsync(string prompt, CancellationToken ct = default)
    {
        if (_client is null)
            throw new InvalidOperationException("Plugin not initialized.");

        var response = await _client.GetResponseAsync(prompt, _options, cancellationToken: ct);
        return response.Text;
    }

    protected override ValueTask OnDisposeAsync()
    {
        _client?.Dispose();
        _client = null;
        return ValueTask.CompletedTask;
    }
}