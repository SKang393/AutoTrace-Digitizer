// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using GraphReader.Inference;
using GraphReader.Markers.Detection;

namespace GraphReader.App.Integration.Workflow;

internal interface IProductionArtifactMaskInferenceRunner
{
    ValueTask<InferenceResponse> RunAsync(
        InferenceRequest request,
        CancellationToken cancellationToken);
}

internal sealed class ProductionArtifactMaskInferenceRunner(Func<InferenceRuntime> runtimeFactory)
    : IProductionArtifactMaskInferenceRunner
{
    private readonly Lazy<InferenceRuntime> runtime = new(
        runtimeFactory ?? throw new ArgumentNullException(nameof(runtimeFactory)),
        LazyThreadSafetyMode.ExecutionAndPublication);

    public ValueTask<InferenceResponse> RunAsync(
        InferenceRequest request,
        CancellationToken cancellationToken) =>
        runtime.Value.RunAsync(request, cancellationToken);
}

internal sealed record ProductionArtifactMaskTensorContract(
    string InputName,
    string OutputName,
    int TensorWidth,
    int TensorHeight,
    int ArtifactChannelIndex,
    string StageVersion,
    TimeSpan Timeout);

internal sealed record ProductionArtifactMaskGateEvidence(
    string ModelSha256,
    string DatasetManifestSha256,
    string EvaluatorSourceSha256,
    string SplitSealSha256,
    int FixtureCount,
    int ExactFixtureCount,
    int DownstreamFalsePositiveCount,
    int DownstreamFalseNegativeCount,
    int DownstreamDuplicateCount);

internal sealed record ProductionArtifactMaskEmbeddedResource(
    string Sha256,
    byte[] Bytes);

/// <summary>
/// Uses the independently gated dense artifact head of the checksum-resolved
/// marker-center model. The adapter is not created unless the same packaged
/// benchmark evidence proves full-frame, seed-only artifact masking with exact
/// downstream marker counts and zero prohibited-structure hits.
/// </summary>
public sealed class ProductionMarkerArtifactMaskAdapter : IProductionArtifactMaskAdapter
{
    public const string ApprovalBenchmarkProfile = "marker-center-artifact-mask-public-gate-v1";
    public const string SeedMaskScope = "ocr_axis_tick_divider_ambiguous_only";
    internal const string FrozenEvaluatorSourceSha256 =
        "58c3cd9ec6bf2255c4b5d2cc9d865e2d37f58b0688a8bceda78ff523398a9f5f";
    private static readonly string[] RequiredInputChannels =
        ["ink_probability", "text_mask", "artifact_mask"];
    private static readonly string[] RequiredOutputChannels =
        ["center_probability", "radius_pixels", "artifact_probability"];
    private static readonly string[] ProhibitedStructureKinds =
        ["text", "axis", "tick", "divider", "bracket", "arrow_shaft", "arrowhead", "legend", "line_intersection"];

    private readonly ProductionArtifactMaskTensorContract tensor;
    private readonly IProductionArtifactMaskInferenceRunner inference;

    internal ProductionMarkerArtifactMaskAdapter(
        ModelIdentity model,
        ProductionArtifactMaskTensorContract tensor,
        bool isApproved,
        IProductionArtifactMaskInferenceRunner inference)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        Model.Validate();
        this.tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
        this.inference = inference ?? throw new ArgumentNullException(nameof(inference));
        IsApproved = isApproved;
    }

    public string AdapterId => $"graphreader-marker-artifact-mask:{Model.Sha256[..12].ToLowerInvariant()}";

    public bool IsApproved { get; }

    public ModelIdentity Model { get; }

    public static ProductionMarkerArtifactMaskAdapter Create(
        ResolvedProductionModel resolvedModel,
        ProductionInferenceRuntimeHost runtimeHost)
    {
        ArgumentNullException.ThrowIfNull(resolvedModel);
        ArgumentNullException.ThrowIfNull(runtimeHost);
        if (!string.Equals(resolvedModel.Task, "marker_center", StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Resolved model task '{resolvedModel.Task}' is not marker_center.");
        }

        if (!resolvedModel.AvailableProviders.Contains(InferenceProvider.Cpu))
        {
            throw new InvalidDataException(
                "The marker artifact-mask model lacks mandatory CPU provider approval.");
        }

        VerifyChecksum(
            resolvedModel.ManifestPath,
            resolvedModel.ManifestSha256,
            "marker artifact-mask manifest");
        VerifyChecksum(
            resolvedModel.BenchmarkEvidencePath,
            resolvedModel.BenchmarkEvidenceSha256,
            "marker artifact-mask benchmark evidence");
        ProductionArtifactMaskTensorContract tensor = ReadContract(
            resolvedModel.ManifestPath,
            resolvedModel.BenchmarkEvidencePath,
            resolvedModel.Identity.Version,
            resolvedModel.BenchmarkEvidenceSha256,
            resolvedModel.Identity.Sha256);
        return new ProductionMarkerArtifactMaskAdapter(
            resolvedModel.Identity,
            tensor,
            isApproved: true,
            new ProductionArtifactMaskInferenceRunner(() => runtimeHost.Runtime));
    }

    public async Task<ProductionArtifactMaskEvidence> DetectAsync(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster raster,
        ProductionDetectionMaskSeed seed,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(raster);
        ArgumentNullException.ThrowIfNull(seed);
        cancellationToken.ThrowIfCancellationRequested();
        if (!IsApproved)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionModelsUnavailable,
                "Errors.ModelNotFound",
                $"Artifact-mask adapter '{AdapterId}' is not production-approved.",
                "Install the exact checksum-bound model with direct artifact-mask gate evidence or continue in manual mode.");
        }

        ValidateInputs(request, raster, seed);
        var total = Stopwatch.StartNew();
        var preprocessing = Stopwatch.StartNew();
        MarkerImageFrame frame = seed.CreateMarkerFrame(raster);
        float[] input = CreateInputTensor(frame, cancellationToken);
        preprocessing.Stop();
        var inferenceRequest = new InferenceRequest(
            Model,
            new InferenceInput(
                input,
                [1, 3, tensor.TensorHeight, tensor.TensorWidth],
                tensor.InputName,
                tensor.OutputName),
            new StageCacheMaterial(
                request.Image.Sha256,
                $"full-frame:{raster.Width}x{raster.Height}",
                $"original-to-tensor:{tensor.TensorWidth}x{tensor.TensorHeight}",
                "marker_artifact_mask",
                tensor.StageVersion,
                new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["adapter"] = "marker-center-artifact-head-full-frame-v1",
                    ["input_channels"] = string.Join(',', RequiredInputChannels),
                    ["output_channels"] = string.Join(',', RequiredOutputChannels),
                    ["artifact_channel"] = tensor.ArtifactChannelIndex,
                    ["seed_scope"] = SeedMaskScope,
                    ["raster_width"] = raster.Width,
                    ["raster_height"] = raster.Height,
                    ["tensor_width"] = tensor.TensorWidth,
                    ["tensor_height"] = tensor.TensorHeight,
                },
                ContractVersion: 1),
            tensor.Timeout);

        InferenceResponse response;
        try
        {
            response = await inference.RunAsync(inferenceRequest, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                $"Artifact-mask inference failed: {exception.Message}",
                "Retry on CPU or continue with manual marker review.");
        }

        if (!response.Succeeded || response.Execution is null)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                $"Artifact-mask inference returned no executable evidence: {response.Error?.Code ?? "unknown_error"}.",
                "Retry on CPU or continue with manual marker review.");
        }

        var postprocessing = Stopwatch.StartNew();
        float[] mask = DecodeArtifactMask(
            response.Execution.Output,
            raster.Width,
            raster.Height,
            seed.ArtifactMaskValues.Span,
            cancellationToken);
        double confidence = ComputeConfidence(mask);
        postprocessing.Stop();
        total.Stop();

        string provider = response.Execution.Provider switch
        {
            InferenceProvider.Cpu => "cpu",
            InferenceProvider.DirectMl => "directml",
            _ => throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Artifact-mask inference used a non-production provider.",
                "Retry with the approved CPU or DirectML provider."),
        };
        string[] warnings = response.ProviderAttempts
            .Where(static attempt => !attempt.Succeeded)
            .Select(attempt => $"provider_attempt_failed:{attempt.Provider.ToString().ToLowerInvariant()}")
            .Append("artifact_mask_scope:marker_center_artifact_head_full_frame_seeded_by_ocr_axis")
            .ToArray();
        var envelope = new WorkflowVisionEnvelope(
            contractVersion: 1,
            request.RunId,
            request.ProjectId,
            request.Panel.ImportedPanel.PanelId,
            stage: "markers",
            tensor.StageVersion,
            request.Image.Sha256,
            new WorkflowVisionModel(Model.ModelId, Model.Version, Model.Sha256, provider),
            new WorkflowVisionTiming(
                preprocessing.Elapsed.TotalMilliseconds,
                response.Execution.Timing.InferenceMilliseconds,
                postprocessing.Elapsed.TotalMilliseconds,
                total.Elapsed.TotalMilliseconds),
            confidence,
            warnings,
            request.Transforms);
        return new ProductionArtifactMaskEvidence(
            raster.Width,
            raster.Height,
            raster.InputSha256,
            raster.Variant,
            envelope,
            mask,
            warnings);
    }

    private float[] CreateInputTensor(
        MarkerImageFrame frame,
        CancellationToken cancellationToken)
    {
        int tensorPixels = checked(tensor.TensorWidth * tensor.TensorHeight);
        var values = new float[checked(tensorPixels * 3)];
        ReadOnlySpan<float> luminance = frame.ChannelsFirstPixels.Span;
        ReadOnlySpan<float> ocrMask = frame.OcrMask.Values.Span;
        ReadOnlySpan<float> artifactMask = frame.ArtifactMask.Values.Span;
        for (int y = 0; y < tensor.TensorHeight; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double sourceY = (((y + 0.5) * frame.Height / tensor.TensorHeight) - 0.5);
            for (int x = 0; x < tensor.TensorWidth; x++)
            {
                double sourceX = (((x + 0.5) * frame.Width / tensor.TensorWidth) - 0.5);
                int index = (y * tensor.TensorWidth) + x;
                values[index] = 1 - Sample(luminance, frame.Width, frame.Height, sourceX, sourceY);
                values[tensorPixels + index] = Sample(ocrMask, frame.Width, frame.Height, sourceX, sourceY);
                values[(2 * tensorPixels) + index] = Sample(
                    artifactMask,
                    frame.Width,
                    frame.Height,
                    sourceX,
                    sourceY);
            }
        }

        return values;
    }

    private float[] DecodeArtifactMask(
        IReadOnlyList<float> output,
        int rasterWidth,
        int rasterHeight,
        ReadOnlySpan<float> seededArtifactMask,
        CancellationToken cancellationToken)
    {
        int tensorPixels = checked(tensor.TensorWidth * tensor.TensorHeight);
        int expectedLength = checked(tensorPixels * 3);
        if (output.Count != expectedLength)
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                $"Artifact-mask model returned {output.Count} values; {expectedLength} are required.",
                "Select the exact model bound to the reviewed tensor contract.");
        }

        var outputValues = output as float[] ?? output.ToArray();
        ValidateOutput(outputValues, tensorPixels, cancellationToken);
        ReadOnlySpan<float> artifactPlane = outputValues.AsSpan(
            tensor.ArtifactChannelIndex * tensorPixels,
            tensorPixels);
        var mask = new float[checked(rasterWidth * rasterHeight)];
        for (int y = 0; y < rasterHeight; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double tensorY = (((y + 0.5) * tensor.TensorHeight / rasterHeight) - 0.5);
            int row = y * rasterWidth;
            for (int x = 0; x < rasterWidth; x++)
            {
                double tensorX = (((x + 0.5) * tensor.TensorWidth / rasterWidth) - 0.5);
                int index = row + x;
                mask[index] = Math.Max(
                    Sample(artifactPlane, tensor.TensorWidth, tensor.TensorHeight, tensorX, tensorY),
                    seededArtifactMask[index]);
            }
        }

        return mask;
    }

    private static void ValidateOutput(
        ReadOnlySpan<float> output,
        int tensorPixels,
        CancellationToken cancellationToken)
    {
        for (int index = 0; index < tensorPixels; index++)
        {
            if ((index & 0x3fff) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            float center = output[index];
            float radius = output[tensorPixels + index];
            float artifact = output[(2 * tensorPixels) + index];
            if (!float.IsFinite(center) || center is < 0 or > 1 ||
                !float.IsFinite(radius) || radius < 0 ||
                !float.IsFinite(artifact) || artifact is < 0 or > 1)
            {
                throw Failure(
                    ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                    "Errors.DetectionEvidenceRejected",
                    "Artifact-mask model output contains invalid center, radius, or artifact values.",
                    "Select the exact activated three-head marker-center model.");
            }
        }
    }

    private static float Sample(
        ReadOnlySpan<float> values,
        int width,
        int height,
        double x,
        double y)
    {
        double clampedX = Math.Clamp(x, 0, width - 1d);
        double clampedY = Math.Clamp(y, 0, height - 1d);
        int x0 = (int)Math.Floor(clampedX);
        int y0 = (int)Math.Floor(clampedY);
        int x1 = Math.Min(width - 1, x0 + 1);
        int y1 = Math.Min(height - 1, y0 + 1);
        double xWeight = clampedX - x0;
        double yWeight = clampedY - y0;
        double top = (values[(y0 * width) + x0] * (1 - xWeight)) +
                     (values[(y0 * width) + x1] * xWeight);
        double bottom = (values[(y1 * width) + x0] * (1 - xWeight)) +
                        (values[(y1 * width) + x1] * xWeight);
        return (float)((top * (1 - yWeight)) + (bottom * yWeight));
    }

    private static double ComputeConfidence(ReadOnlySpan<float> mask)
    {
        if (mask.IsEmpty)
        {
            return 0;
        }

        double certainty = 0;
        foreach (float value in mask)
        {
            certainty += 2 * Math.Abs(value - 0.5);
        }

        return Math.Clamp(certainty / mask.Length, 0, 1);
    }

    private static void ValidateInputs(
        ProductionWorkflowDetectionRequest request,
        ProductionDecodedRaster raster,
        ProductionDetectionMaskSeed seed)
    {
        if (request.ImageVariant != WorkflowImageVariant.Original ||
            raster.Variant != WorkflowImageVariant.Original ||
            seed.RasterVariant != WorkflowImageVariant.Original ||
            raster.Width != request.Image.Width || raster.Height != request.Image.Height ||
            seed.Width != raster.Width || seed.Height != raster.Height ||
            raster.OriginalToFrame != MarkerAffineTransform.Identity ||
            !string.Equals(request.Image.Sha256, raster.InputSha256, StringComparison.OrdinalIgnoreCase) ||
            !string.Equals(seed.RasterSha256, raster.InputSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw Failure(
                ProductionWorkflowFailureCodes.DetectionEvidenceRejected,
                "Errors.DetectionEvidenceRejected",
                "Artifact-mask inference requires current immutable original-raster seed evidence.",
                "Rebuild the seed masks from the retained original image.");
        }
    }

    private static ProductionArtifactMaskTensorContract ReadContract(
        string manifestPath,
        string benchmarkEvidencePath,
        string modelVersion,
        string packagedBenchmarkSha256,
        string modelSha256)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        JsonElement root = document.RootElement;
        JsonElement input = SingleObject(root, "inputs");
        JsonElement output = SingleObject(root, "outputs");
        RequireString(input, "element_type", "float32");
        RequireString(input, "layout", "NCHW");
        RequireString(output, "element_type", "float32");
        RequireString(output, "layout", "NCHW");
        RequireDynamicShape(input, "shape");
        RequireDynamicShape(output, "shape");
        RequireStringArray(input, "channels", RequiredInputChannels);
        RequireStringArray(output, "channels", RequiredOutputChannels);
        if (RequiredInt32(output, "output_stride") != 1)
        {
            throw new InvalidDataException("Artifact-mask output stride must be one.");
        }

        JsonElement preprocessing = RequiredObject(root, "preprocessing");
        if (RequiredSingle(preprocessing, "normalization_mean") != 0 ||
            RequiredSingle(preprocessing, "normalization_scale") != 1)
        {
            throw new InvalidDataException(
                "Artifact-mask input must retain unnormalized [0,1] planes.");
        }

        JsonElement providerEvidence = SingleBenchmark(
            root,
            "graphreader-inference-cpu-directml-parity");
        int[] inputShape = ReadFixedShape(providerEvidence, "input_shape");
        int[] outputShape = ReadFixedShape(providerEvidence, "output_shape");
        if (inputShape.Length != 4 || outputShape.Length != 4 ||
            inputShape[0] != 1 || inputShape[1] != 3 ||
            outputShape[0] != 1 || outputShape[1] != 3 ||
            inputShape[2] != outputShape[2] || inputShape[3] != outputShape[3] ||
            inputShape[2] is < 32 or > 2048 || inputShape[3] is < 32 or > 2048)
        {
            throw new InvalidDataException(
                "Artifact-mask provider evidence must bind equal bounded NCHW [1,3,H,W] shapes.");
        }

        ProductionArtifactMaskGateEvidence directEvidence = ReadDirectGateEvidence(
            benchmarkEvidencePath,
            modelSha256);
        JsonElement approval = SingleBenchmark(root, ApprovalBenchmarkProfile);
        RequireBoolean(approval, "release_eligible", expected: true);
        RequireBoolean(approval, "production_approval", expected: true);
        RequireBoolean(approval, "private_data", expected: false);
        RequireBoolean(approval, "chandler_used", expected: false);
        RequireString(approval, "seed_mask_scope", SeedMaskScope);
        RequireString(approval, "coordinate_space", "original_pixels");
        RequireString(approval, "provider", "cpu");
        RequireMatchingSha256(approval, "model_sha256", directEvidence.ModelSha256);
        RequireMatchingSha256(
            approval,
            "dataset_manifest_sha256",
            directEvidence.DatasetManifestSha256);
        RequireMatchingSha256(
            approval,
            "evaluator_source_sha256",
            directEvidence.EvaluatorSourceSha256);
        RequireMatchingSha256(approval, "split_seal_sha256", directEvidence.SplitSealSha256);
        int fixtureCount = RequiredInt32(approval, "fixture_count");
        int exactFixtureCount = RequiredInt32(approval, "exact_fixture_count");
        if (fixtureCount != directEvidence.FixtureCount ||
            exactFixtureCount != directEvidence.ExactFixtureCount ||
            RequiredInt32(approval, "downstream_false_positive_count") !=
                directEvidence.DownstreamFalsePositiveCount ||
            RequiredInt32(approval, "downstream_false_negative_count") !=
                directEvidence.DownstreamFalseNegativeCount ||
            RequiredInt32(approval, "downstream_duplicate_count") !=
                directEvidence.DownstreamDuplicateCount)
        {
            throw new InvalidDataException(
                "Artifact-mask manifest metrics do not match the checksum-bound direct benchmark report.");
        }

        JsonElement prohibitedHits = RequiredObject(approval, "prohibited_structure_hits");
        foreach (string kind in ProhibitedStructureKinds)
        {
            if (RequiredInt32(prohibitedHits, kind) != 0)
            {
                throw new InvalidDataException(
                    $"Artifact-mask approval contains a prohibited '{kind}' marker hit.");
            }
        }

        string evidenceSha256 = RequiredString(approval, "evidence_sha256");
        if (!IsSha256(evidenceSha256) ||
            !string.Equals(evidenceSha256, packagedBenchmarkSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                "Artifact-mask approval must be the checksum-bound benchmark evidence selected by the production model store.");
        }

        return new ProductionArtifactMaskTensorContract(
            RequiredString(input, "name"),
            RequiredString(output, "name"),
            inputShape[3],
            inputShape[2],
            ArtifactChannelIndex: 2,
            StageVersion: $"marker-artifact-mask-v1:{modelVersion}",
            Timeout: TimeSpan.FromSeconds(30));
    }

    private static ProductionArtifactMaskGateEvidence ReadDirectGateEvidence(
        string evidencePath,
        string modelSha256)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(evidencePath));
        JsonElement root = document.RootElement;
        RequireString(root, "schema", "graphreader.marker-artifact-mask-gate.v1");
        RequireString(root, "profile", ApprovalBenchmarkProfile);
        RequireString(root, "status", "pass");
        RequireString(root, "scope", "public_synthetic_sealed");
        RequireString(root, "provider", "cpu");
        RequireString(root, "seed_mask_scope", SeedMaskScope);
        RequireString(root, "coordinate_space", "original_pixels");
        RequireBoolean(root, "release_eligible", expected: true);
        RequireBoolean(root, "production_approval", expected: true);
        RequireBoolean(root, "private_data", expected: false);
        RequireBoolean(root, "chandler_used", expected: false);
        RequireMatchingSha256(root, "model_sha256", modelSha256);

        JsonElement resources = RequiredObject(root, "reviewed_resources");
        ProductionArtifactMaskEmbeddedResource datasetManifest = ValidateEmbeddedResource(
            resources,
            "dataset_manifest",
            "application/json");
        ProductionArtifactMaskEmbeddedResource evaluatorSource = ValidateEmbeddedResource(
            resources,
            "evaluator_source",
            "text/x-python");
        ProductionArtifactMaskEmbeddedResource splitSeal = ValidateEmbeddedResource(
            resources,
            "split_seal",
            "application/json");
        if (!string.Equals(
                evaluatorSource.Sha256,
                FrozenEvaluatorSourceSha256,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                "Direct artifact-mask evidence does not contain the frozen reviewed evaluator source.");
        }

        HashSet<string> datasetFixtureIds = ReadDatasetFixtureIds(datasetManifest.Bytes);
        ValidateSplitSeal(
            splitSeal.Bytes,
            datasetManifest.Sha256,
            evaluatorSource.Sha256,
            datasetFixtureIds);

        int fixtureCount = RequiredInt32(root, "fixture_count");
        int exactFixtureCount = RequiredInt32(root, "exact_fixture_count");
        int falsePositiveCount = RequiredInt32(root, "downstream_false_positive_count");
        int falseNegativeCount = RequiredInt32(root, "downstream_false_negative_count");
        int duplicateCount = RequiredInt32(root, "downstream_duplicate_count");
        if (fixtureCount < 3 || exactFixtureCount != fixtureCount ||
            falsePositiveCount != 0 || falseNegativeCount != 0 || duplicateCount != 0)
        {
            throw new InvalidDataException(
                "Direct artifact-mask evidence requires at least three exact fixtures with zero false positives, false negatives, and duplicates.");
        }

        RequireZeroProhibitedHits(RequiredObject(root, "prohibited_structure_hits"));
        JsonElement[] fixtureResults = RequiredArray(root, "fixture_results").EnumerateArray().ToArray();
        if (fixtureResults.Length != fixtureCount)
        {
            throw new InvalidDataException(
                "Direct artifact-mask fixture results do not match the declared fixture count.");
        }

        var fixtureIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonElement fixture in fixtureResults)
        {
            if (fixture.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("Each direct artifact-mask fixture result must be an object.");
            }

            string fixtureId = RequiredString(fixture, "fixture_id");
            RequireBoolean(fixture, "exact_count", expected: true);
            if (!fixtureIds.Add(fixtureId) ||
                RequiredInt32(fixture, "false_positive_count") != 0 ||
                RequiredInt32(fixture, "false_negative_count") != 0 ||
                RequiredInt32(fixture, "duplicate_count") != 0)
            {
                throw new InvalidDataException(
                    "Direct artifact-mask fixture results require unique IDs, exact counts, and zero errors.");
            }

            RequireZeroProhibitedHits(RequiredObject(fixture, "prohibited_structure_hits"));
        }

        if (!fixtureIds.SetEquals(datasetFixtureIds))
        {
            throw new InvalidDataException(
                "Direct artifact-mask report fixture IDs do not match the frozen sealed dataset split.");
        }

        return new ProductionArtifactMaskGateEvidence(
            modelSha256.ToLowerInvariant(),
            datasetManifest.Sha256,
            evaluatorSource.Sha256,
            splitSeal.Sha256,
            fixtureCount,
            exactFixtureCount,
            falsePositiveCount,
            falseNegativeCount,
            duplicateCount);
    }

    private static ProductionArtifactMaskEmbeddedResource ValidateEmbeddedResource(
        JsonElement resources,
        string resourceName,
        string expectedMediaType)
    {
        const int MaximumDecodedBytes = 4 * 1024 * 1024;
        JsonElement resource = RequiredObject(resources, resourceName);
        RequireString(resource, "media_type", expectedMediaType);
        RequireString(resource, "encoding", "base64");
        string expectedSha256 = RequireSha256(resource, "sha256");
        string encoded = RequiredString(resource, "content_base64");
        if (encoded.Length > ((MaximumDecodedBytes + 2) / 3 * 4) + 4)
        {
            throw new InvalidDataException(
                $"Embedded artifact-mask resource '{resourceName}' exceeds the size limit.");
        }

        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(encoded);
        }
        catch (FormatException exception)
        {
            throw new InvalidDataException(
                $"Embedded artifact-mask resource '{resourceName}' is not valid base64.",
                exception);
        }

        if (bytes.Length == 0 || bytes.Length > MaximumDecodedBytes)
        {
            throw new InvalidDataException(
                $"Embedded artifact-mask resource '{resourceName}' has an invalid size.");
        }

        string actualSha256 = Convert.ToHexStringLower(SHA256.HashData(bytes));
        if (!string.Equals(actualSha256, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Embedded artifact-mask resource '{resourceName}' does not match its SHA-256.");
        }

        return new ProductionArtifactMaskEmbeddedResource(actualSha256, bytes);
    }

    private static HashSet<string> ReadDatasetFixtureIds(byte[] datasetManifestBytes)
    {
        using JsonDocument document = JsonDocument.Parse(datasetManifestBytes);
        JsonElement root = document.RootElement;
        RequireString(root, "schema", "graphreader.marker-artifact-mask-dataset.v1");
        RequireString(root, "scope", "public_synthetic");
        RequireBoolean(root, "private_data", expected: false);
        RequireBoolean(root, "chandler_used", expected: false);
        if (RequiredInt32(root, "seed") != 393)
        {
            throw new InvalidDataException("Artifact-mask dataset seed must equal the frozen seed 393.");
        }

        JsonElement[] fixtures = RequiredArray(root, "fixtures").EnumerateArray().ToArray();
        if (fixtures.Length < 3)
        {
            throw new InvalidDataException("Artifact-mask dataset requires at least three sealed fixtures.");
        }

        var fixtureIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonElement fixture in fixtures)
        {
            if (fixture.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("Each artifact-mask dataset fixture must be an object.");
            }

            string fixtureId = RequiredString(fixture, "fixture_id");
            if (!fixtureIds.Add(fixtureId))
            {
                throw new InvalidDataException("Artifact-mask dataset fixture IDs must be unique.");
            }

            RequireSha256(fixture, "image_sha256");
            RequireSha256(fixture, "ground_truth_sha256");
            RequiredString(fixture, "family");
        }

        return fixtureIds;
    }

    private static void ValidateSplitSeal(
        byte[] splitSealBytes,
        string datasetManifestSha256,
        string evaluatorSourceSha256,
        HashSet<string> datasetFixtureIds)
    {
        using JsonDocument document = JsonDocument.Parse(splitSealBytes);
        JsonElement root = document.RootElement;
        RequireString(root, "schema", "graphreader.marker-artifact-mask-split-seal.v1");
        RequireString(root, "profile", ApprovalBenchmarkProfile);
        RequireBoolean(root, "sealed", expected: true);
        RequireBoolean(root, "selection_locked_before_inference", expected: true);
        RequireBoolean(root, "private_data", expected: false);
        RequireBoolean(root, "chandler_used", expected: false);
        RequireMatchingSha256(root, "dataset_manifest_sha256", datasetManifestSha256);
        RequireMatchingSha256(root, "evaluator_source_sha256", evaluatorSourceSha256);
        if (RequiredInt32(root, "fixture_count") != datasetFixtureIds.Count)
        {
            throw new InvalidDataException(
                "Artifact-mask split seal fixture count does not match the dataset manifest.");
        }

        var splitFixtureIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonElement value in RequiredArray(root, "fixture_ids").EnumerateArray())
        {
            if (value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(value.GetString()) ||
                !splitFixtureIds.Add(value.GetString()!))
            {
                throw new InvalidDataException(
                    "Artifact-mask split seal fixture IDs must be unique nonempty strings.");
            }
        }

        if (!splitFixtureIds.SetEquals(datasetFixtureIds))
        {
            throw new InvalidDataException(
                "Artifact-mask split seal fixture IDs do not match the dataset manifest.");
        }
    }

    private static void RequireZeroProhibitedHits(JsonElement prohibitedHits)
    {
        foreach (string kind in ProhibitedStructureKinds)
        {
            if (RequiredInt32(prohibitedHits, kind) != 0)
            {
                throw new InvalidDataException(
                    $"Artifact-mask approval contains a prohibited '{kind}' marker hit.");
            }
        }
    }

    private static JsonElement SingleBenchmark(JsonElement root, string profile)
    {
        JsonElement[] matches = RequiredArray(root, "benchmarks").EnumerateArray()
            .Where(element =>
                element.ValueKind == JsonValueKind.Object &&
                element.TryGetProperty("profile", out JsonElement value) &&
                value.ValueKind == JsonValueKind.String &&
                string.Equals(value.GetString(), profile, StringComparison.Ordinal) &&
                element.TryGetProperty("status", out JsonElement status) &&
                status.ValueKind == JsonValueKind.String &&
                string.Equals(status.GetString(), "pass", StringComparison.Ordinal))
            .ToArray();
        if (matches.Length != 1)
        {
            throw new InvalidDataException(
                $"Marker-center manifest must contain one passing '{profile}' benchmark.");
        }

        return matches[0];
    }

    private static JsonElement SingleObject(JsonElement root, string propertyName)
    {
        JsonElement array = RequiredArray(root, propertyName);
        JsonElement[] values = array.EnumerateArray().ToArray();
        if (values.Length != 1 || values[0].ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"Manifest '{propertyName}' must contain exactly one object.");
        }

        return values[0];
    }

    private static JsonElement RequiredArray(JsonElement parent, string propertyName)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidDataException($"Manifest array '{propertyName}' is required.");
        }

        return value;
    }

    private static JsonElement RequiredObject(JsonElement parent, string propertyName)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"Manifest object '{propertyName}' is required.");
        }

        return value;
    }

    private static string RequiredString(JsonElement parent, string propertyName)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(value.GetString()))
        {
            throw new InvalidDataException($"Manifest string '{propertyName}' is required.");
        }

        return value.GetString()!;
    }

    private static void RequireString(JsonElement parent, string propertyName, string expected)
    {
        if (!string.Equals(RequiredString(parent, propertyName), expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Manifest '{propertyName}' must equal '{expected}'.");
        }
    }

    private static void RequireStringArray(
        JsonElement parent,
        string propertyName,
        IReadOnlyList<string> expected)
    {
        JsonElement array = RequiredArray(parent, propertyName);
        string[] actual;
        try
        {
            actual = array.EnumerateArray().Select(static value => value.GetString()!).ToArray();
        }
        catch (InvalidOperationException exception)
        {
            throw new InvalidDataException(
                $"Manifest array '{propertyName}' must contain strings.",
                exception);
        }

        if (!actual.SequenceEqual(expected, StringComparer.Ordinal))
        {
            throw new InvalidDataException(
                $"Manifest array '{propertyName}' does not match the production tensor contract.");
        }
    }

    private static int RequiredInt32(JsonElement parent, string propertyName)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int result))
        {
            throw new InvalidDataException($"Manifest integer '{propertyName}' is required.");
        }

        return result;
    }

    private static int[] ReadFixedShape(JsonElement parent, string propertyName)
    {
        try
        {
            return RequiredArray(parent, propertyName)
                .EnumerateArray()
                .Select(static value => value.GetInt32())
                .ToArray();
        }
        catch (Exception exception) when (exception is InvalidOperationException or FormatException)
        {
            throw new InvalidDataException(
                $"Manifest shape '{propertyName}' must contain integers.",
                exception);
        }
    }

    private static void RequireDynamicShape(JsonElement parent, string propertyName)
    {
        JsonElement[] values = RequiredArray(parent, propertyName).EnumerateArray().ToArray();
        if (values.Length != 4 ||
            values[0].ValueKind != JsonValueKind.String || values[0].GetString() != "N" ||
            values[1].ValueKind != JsonValueKind.Number || !values[1].TryGetInt32(out int channels) || channels != 3 ||
            values[2].ValueKind != JsonValueKind.String || values[2].GetString() != "H" ||
            values[3].ValueKind != JsonValueKind.String || values[3].GetString() != "W")
        {
            throw new InvalidDataException(
                $"Manifest '{propertyName}' must declare dynamic NCHW [N,3,H,W].");
        }
    }

    private static float RequiredSingle(JsonElement parent, string propertyName)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float result) ||
            !float.IsFinite(result))
        {
            throw new InvalidDataException($"Manifest finite float '{propertyName}' is required.");
        }

        return result;
    }

    private static void RequireBoolean(JsonElement parent, string propertyName, bool expected)
    {
        if (!parent.TryGetProperty(propertyName, out JsonElement value) ||
            value.ValueKind != (expected ? JsonValueKind.True : JsonValueKind.False))
        {
            throw new InvalidDataException($"Manifest boolean '{propertyName}' must be {expected.ToString().ToLowerInvariant()}.");
        }
    }

    private static string RequireSha256(JsonElement parent, string propertyName)
    {
        string value = RequiredString(parent, propertyName);
        if (!IsSha256(value))
        {
            throw new InvalidDataException($"Manifest SHA-256 '{propertyName}' is invalid.");
        }

        return value;
    }

    private static void RequireMatchingSha256(
        JsonElement parent,
        string propertyName,
        string expected)
    {
        string actual = RequireSha256(parent, propertyName);
        if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Manifest SHA-256 '{propertyName}' does not match the packaged model.");
        }
    }

    private static void VerifyChecksum(string path, string expectedSha256, string label)
    {
        string actual = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"The checksum-resolved {label} changed after model-store validation.");
        }
    }

    private static bool IsSha256(string value) =>
        value.Length == 64 && value.All(Uri.IsHexDigit);

    private static ProductionWorkflowStageException Failure(
        string code,
        string userMessageKey,
        string technicalMessage,
        string suggestedAction) =>
        new(new ProductionWorkflowFailure(
            code,
            userMessageKey,
            technicalMessage,
            Recoverable: true,
            suggestedAction));
}
