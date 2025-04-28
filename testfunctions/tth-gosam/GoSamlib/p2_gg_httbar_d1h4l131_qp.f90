module     p2_gg_httbar_d1h4l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d1h4l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1mu0 = 0
   integer, parameter :: ninjaidxt0mu0 = 1
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd1h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd1
      complex(ki), dimension (0:*), intent(inout) :: brack
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd1h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(26) :: acd1
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd1(1)=dotproduct(k2,ninjaE3)
      acd1(2)=abb1(21)
      acd1(3)=dotproduct(ninjaE3,spvae2e1)
      acd1(4)=abb1(7)
      acd1(5)=dotproduct(ninjaE3,spvae1e2)
      acd1(6)=abb1(8)
      acd1(7)=dotproduct(ninjaE3,spval5l4)
      acd1(8)=abb1(9)
      acd1(9)=dotproduct(ninjaE3,spval5l3)
      acd1(10)=abb1(11)
      acd1(11)=dotproduct(ninjaE3,spval3k2)
      acd1(12)=abb1(12)
      acd1(13)=dotproduct(k2,ninjaA)
      acd1(14)=dotproduct(ninjaA,spvae2e1)
      acd1(15)=dotproduct(ninjaA,spvae1e2)
      acd1(16)=dotproduct(ninjaA,spval5l4)
      acd1(17)=dotproduct(ninjaA,spval5l3)
      acd1(18)=dotproduct(ninjaA,spval3k2)
      acd1(19)=abb1(10)
      acd1(20)=acd1(1)*acd1(2)
      acd1(21)=acd1(3)*acd1(4)
      acd1(22)=acd1(5)*acd1(6)
      acd1(23)=acd1(7)*acd1(8)
      acd1(24)=acd1(9)*acd1(10)
      acd1(25)=acd1(11)*acd1(12)
      acd1(20)=acd1(25)+acd1(24)+acd1(23)+acd1(22)+acd1(20)+acd1(21)
      acd1(21)=acd1(13)*acd1(2)
      acd1(22)=acd1(14)*acd1(4)
      acd1(23)=acd1(15)*acd1(6)
      acd1(24)=acd1(16)*acd1(8)
      acd1(25)=acd1(17)*acd1(10)
      acd1(26)=acd1(18)*acd1(12)
      acd1(21)=acd1(19)+acd1(26)+acd1(25)+acd1(24)+acd1(23)+acd1(21)+acd1(22)
      brack(ninjaidxt1mu0)=acd1(20)
      brack(ninjaidxt0mu0)=acd1(21)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d1h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd1h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-2))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d1h4l131_qp
