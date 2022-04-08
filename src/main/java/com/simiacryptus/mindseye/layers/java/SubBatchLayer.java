/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.WrapperLayer;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * The SubBatchLayer class represents a layer in a neural network that
 * consists of a number of sub-batches.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class SubBatchLayer extends WrapperLayer {

  /**
   * Instantiates a new Sub batch layer.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected SubBatchLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  /**
   * Instantiates a new Sub batch layer.
   *
   * @param inner the inner
   */
  public SubBatchLayer(final Layer inner) {
    super(inner);
  }

  @Override
  public RefList<Layer> getChildren() {
    return super.getChildren();
  }

  /**
   * Creates a new SubBatchLayer from the given JSON object.
   *
   * @param json The JSON object to create the SubBatchLayer from.
   * @param rs   A map containing the character sequences and byte arrays.
   * @return The newly created SubBatchLayer.
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static SubBatchLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SubBatchLayer(json, rs);
  }

  /**
   * Returns a new SubBatchLayer that wraps the specified layer.
   *
   * @param layer the layer to wrap
   * @return a new SubBatchLayer
   * @docgenVersion 9
   */
  @Nonnull
  public static <T extends Layer> SubBatchLayer wrap(@Nullable T layer) {
    return new SubBatchLayer(layer);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inputs) {
    Layer inner = getInner();
    TensorList data0 = inputs[0].getData();
    int batches = data0.length();
    data0.freeRef();
    Tensor[][] passbackBuffer = RefIntStream.range(0, inputs.length)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor[]>) inputIndex -> {
          TensorList data = inputs[inputIndex].getData();
          Tensor[] tensors = new Tensor[data.length()];
          data.freeRef();
          return tensors;
        }, RefUtil.addRef(inputs))).toArray(Tensor[][]::new);
    Result[] batchResults = RefIntStream.range(0, batches)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Result>) batchIndex -> {
          assert inner != null;
          return inner.eval(RefIntStream.range(0, inputs.length)
              .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Result>) inputIndex -> {
                Tensor[] tensors = RefUtil.addRef(passbackBuffer[inputIndex]);
                Result.Accumulator accumulator = new SubAccumulator(tensors, batchIndex);
                TensorList tensorList = inputs[inputIndex].getData();
                Result result = new Result(new TensorArray(tensorList.get(batchIndex)), accumulator);
                tensorList.freeRef();
                return result;
              }, RefUtil.addRef(passbackBuffer), RefUtil.addRef(inputs))).<Result>toArray(Result[]::new));
        }, RefUtil.addRef(passbackBuffer), RefUtil.addRef(inputs), inner == null ? null : inner.addRef()))
        .toArray(Result[]::new);
    if (null != inner)
      inner.freeRef();
    TensorArray resultData = new TensorArray(RefArrays.stream(RefUtil.addRef(batchResults)).map(result -> {
      TensorList tensorList = result.getData();
      result.freeRef();
      Tensor tensor = tensorList.get(0);
      tensorList.freeRef();
      return tensor;
    }).toArray(Tensor[]::new));
    return new Result(resultData, new MainAccumulator(batchResults, passbackBuffer, inputs));
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SubBatchLayer addRef() {
    return (SubBatchLayer) super.addRef();
  }

  /**
   * This class represents a SubAccumulator, which is used to store tensors and batch indices.
   *
   * @docgenVersion 9
   */
  private static class SubAccumulator extends Result.Accumulator {

    private final Tensor[] tensors;
    private final int batchIndex;

    /**
     * Instantiates a new Sub accumulator.
     *
     * @param tensors    the tensors
     * @param batchIndex the batch index
     */
    public SubAccumulator(Tensor[] tensors, int batchIndex) {
      this.tensors = tensors;
      this.batchIndex = batchIndex;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> deltaBuffer, @Nonnull TensorList deltaSignal) {
      if (null != deltaBuffer)
        deltaBuffer.freeRef();
      RefUtil.set(tensors, batchIndex, deltaSignal.get(0));
      deltaSignal.freeRef();
    }

    /**
     * Frees resources.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(tensors);
    }
  }

  /**
   * The MainAccumulator class is used to store the results of a batch of computations,
   * as well as the intermediate results that are passed back between passes.
   * It has fields for storing the results of the computations,
   * as well as the inputs to those computations.
   *
   * @docgenVersion 9
   */
  private static class MainAccumulator extends Result.Accumulator {

    private final Result[] batchResults;
    private final Tensor[][] passbackBuffer;
    private final Result[] inputs;

    /**
     * Instantiates a new Main accumulator.
     *
     * @param batchResults   the batch results
     * @param passbackBuffer the passback buffer
     * @param inputs         the inputs
     */
    public MainAccumulator(Result[] batchResults, Tensor[][] passbackBuffer, Result... inputs) {
      this.batchResults = batchResults;
      this.passbackBuffer = passbackBuffer;
      this.inputs = inputs;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> deltaBuffer, @Nonnull TensorList deltaSignal) {
      RefIntStream.range(0, deltaSignal.length()).forEach(RefUtil.wrapInterface(batchIndex -> {
            TensorArray tensorArray = new TensorArray(deltaSignal.get(batchIndex));
            Result.Accumulator temp_10_0012 = batchResults[batchIndex].getAccumulator();
            assert temp_10_0012 != null;
            temp_10_0012.accept(deltaBuffer == null ? null : deltaBuffer.addRef(),
                tensorArray.addRef());
            temp_10_0012.freeRef();
            tensorArray.freeRef();
          }, deltaSignal.addRef(), RefUtil.addRef(batchResults),
          deltaBuffer == null ? null : deltaBuffer.addRef()));
      deltaSignal.freeRef();
      synchronized (passbackBuffer) {
        RefIntStream.range(0, inputs.length).forEach(RefUtil.wrapInterface(inputIndex -> {
              TensorArray tensorArray = new TensorArray(RefUtil.addRef(passbackBuffer[inputIndex]));
              Result.Accumulator temp_10_0013 = inputs[inputIndex].getAccumulator();
              assert temp_10_0013 != null;
              temp_10_0013.accept(deltaBuffer == null ? null : deltaBuffer.addRef(),
                  tensorArray.addRef());
              temp_10_0013.freeRef();
              tensorArray.freeRef();
            }, RefUtil.addRef(passbackBuffer), RefUtil.addRef(inputs),
            deltaBuffer == null ? null : deltaBuffer.addRef()));
      }
      if (null != deltaBuffer)
        deltaBuffer.freeRef();
    }

    /**
     * Frees resources.
     *
     * @docgenVersion 9
     */
    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inputs);
      RefUtil.freeRef(passbackBuffer);
      RefUtil.freeRef(batchResults);
    }
  }
}
