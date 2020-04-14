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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefComparator;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;

/**
 * The type Max meta layer.
 */
@SuppressWarnings("serial")
public class MaxMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxMetaLayer.class);

  /**
   * Instantiates a new Max meta layer.
   */
  public MaxMetaLayer() {
  }

  /**
   * Instantiates a new Max meta layer.
   *
   * @param id the id
   */
  protected MaxMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json max meta layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the max meta layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static MaxMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxMetaLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result input = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = input.getData();
    final int itemCnt = inputData.length();
    final Tensor input0Tensor = inputData.get(0);
    inputData.freeRef();
    final int vectorSize = input0Tensor.length();
    @Nonnull final int[] indicies = new int[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
      final int itemNumber = i;
      indicies[i] = RefUtil.get(RefIntStream.range(0, itemCnt).mapToObj(x -> x).max(
          RefComparator.comparingDouble(RefUtil.wrapInterface(dataIndex -> {
            TensorList temp_40_0007 = input.getData();
            Tensor tensor = temp_40_0007.get(dataIndex);
            temp_40_0007.freeRef();
            double temp_40_0003 = tensor.get(itemNumber);
            tensor.freeRef();
            return temp_40_0003;
          }, input.addRef()))));
    }
    Result.Accumulator accumulator = new Accumulator(input0Tensor.addRef(), itemCnt, indicies, input.getAccumulator(), input.isAlive());
    boolean alive = input.isAlive();
    TensorArray data = fwd(input, input0Tensor, indicies);
    return new Result(data, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  MaxMetaLayer addRef() {
    return (MaxMetaLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(Result input, Tensor input0Tensor, int[] indicies) {
    TensorArray tensorArray = new TensorArray(input0Tensor.mapIndex(RefUtil.wrapInterface((v, c) -> {
      TensorList tensorList = input.getData();
      Tensor tensor = tensorList.get(indicies[c]);
      tensorList.freeRef();
      double value = tensor.get(c);
      tensor.freeRef();
      return value;
    }, input)));
    input0Tensor.freeRef();
    return tensorArray;
  }

  private static class Accumulator extends Result.Accumulator {

    private final Tensor input0Tensor;
    private final int itemCnt;
    private final int[] indicies;
    private Result.Accumulator accumulator;
    private boolean alive;

    /**
     * Instantiates a new Accumulator.
     *
     * @param input0Tensor the input 0 tensor
     * @param itemCnt      the item cnt
     * @param indicies     the indicies
     * @param accumulator  the accumulator
     * @param alive        the alive
     */
    public Accumulator(Tensor input0Tensor, int itemCnt, int[] indicies, Result.Accumulator accumulator, boolean alive) {
      this.input0Tensor = input0Tensor;
      this.itemCnt = itemCnt;
      this.indicies = indicies;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
      if (alive) {
        @Nullable final Tensor delta = data.get(0);
        @Nonnull final Tensor feedback[] = new Tensor[itemCnt];
        RefArrays.parallelSetAll(RefUtil.addRef(feedback),
            RefUtil.wrapInterface(i -> new Tensor(delta.getDimensions()), delta.addRef()));
        input0Tensor.coordStream(true)
            .forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) inputCoord -> {
              feedback[indicies[inputCoord.getIndex()]].add(inputCoord, delta.get(inputCoord));
            }, delta.addRef(), RefUtil.addRef(feedback)));
        delta.freeRef();
        @Nonnull
        TensorArray tensorArray = new TensorArray(RefUtil.addRef(feedback));
        RefUtil.freeRef(feedback);
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, tensorArray);
      }
      data.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      input0Tensor.freeRef();
    }
  }
}
