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
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

import static com.simiacryptus.mindseye.lang.Result.anyAlive;

/**
 * The type Product layer.
 */
@SuppressWarnings("serial")
public class ProductLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ProductLayer.class);

  /**
   * Instantiates a new Product layer.
   */
  public ProductLayer() {
  }

  /**
   * Instantiates a new Product layer.
   *
   * @param id the id
   */
  protected ProductLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  /**
   * From json product layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the product layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ProductLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ProductLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    assert RefArrays.stream(RefUtil.addRef(inObj)).mapToInt(x -> {
      TensorList tensorList = x.getData();
      int length = tensorList.length();
      tensorList.freeRef();
      x.freeRef();
      return length;
    }).distinct().count() == 1 : RefArrays.toString(RefArrays.stream(RefUtil.addRef(inObj)).mapToInt(x -> {
      TensorList data = x.getData();
      int length = data.length();
      data.freeRef();
      x.freeRef();
      return length;
    }).toArray());
    TensorList data0 = in0.getData();
    int length0 = data0.length();
    in0.freeRef();
    data0.freeRef();
    @Nonnull final double[] sum_A = new double[length0];
    TensorArray data = fwd(length0, sum_A, RefUtil.addRef(inObj));
    boolean alive = anyAlive(RefUtil.addRef(inObj));
    Accumulator accumulator = new Accumulator(sum_A, inObj);
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
  ProductLayer addRef() {
    return (ProductLayer) super.addRef();
  }

  @NotNull
  private TensorArray fwd(int length0, double[] sum_out, @Nonnull Result[] inObj) {
    final Tensor[] outputA = RefIntStream.range(0, length0)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          double sum = 1;
          for (@Nonnull final Result input : inObj) {
            TensorList tensorList = input.getData();
            Tensor tensor = tensorList.get(dataIndex);
            tensorList.freeRef();
            @Nullable final double[] tensorData = tensor.getData();
            tensor.freeRef();
            for (final double element2 : tensorData) {
              sum *= element2;
            }
          }
          sum_out[dataIndex] = sum;
          return new Tensor(new double[]{sum}, 1);
        }, inObj)).toArray(Tensor[]::new);
    return new TensorArray(outputA);
  }

  private static class Accumulator extends Result.Accumulator {

    private final double[] sum_A;
    private final Result[] inObj;

    /**
     * Instantiates a new Accumulator.
     *
     * @param sum_A the sum a
     * @param inObj the in obj
     */
    public Accumulator(double[] sum_A, Result... inObj) {
      this.sum_A = sum_A;
      this.inObj = inObj;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      for (@Nonnull final Result input : inObj) {
        if (input.isAlive()) {
          TensorList data = input.getData();
          DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
          Result.Accumulator accumulator = input.getAccumulator();
          try {
            accumulator.accept(buffer1, new TensorArray(RefIntStream.range(0, delta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  Tensor dataTensor = delta.get(dataIndex);
                  Tensor lTensor = data.get(dataIndex);
                  @Nonnull final Tensor passback = new Tensor(lTensor.getDimensions());
                  for (int i = 0; i < lTensor.length(); i++) {
                    double d = lTensor.get(i);
                    double deltaV = dataTensor.get(0);
                    final double value = d == 0 ? 0 : deltaV * sum_A[dataIndex] / d;
                    passback.set(i, value);
                  }
                  lTensor.freeRef();
                  dataTensor.freeRef();
                  return passback;
                }, data.addRef(), delta.addRef()))
                .toArray(Tensor[]::new)));
          } finally {
            accumulator.freeRef();
          }
          data.freeRef();
        }
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      RefUtil.freeRef(inObj);
    }
  }
}
