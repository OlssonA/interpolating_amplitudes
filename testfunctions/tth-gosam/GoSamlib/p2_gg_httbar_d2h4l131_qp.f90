module     p2_gg_httbar_d2h4l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d2h4l131_qp.f90
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
      use p2_gg_httbar_abbrevd2h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd2
      complex(ki), dimension (0:*), intent(inout) :: brack
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd2h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(26) :: acd2
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd2(1)=dotproduct(k2,ninjaE3)
      acd2(2)=abb2(21)
      acd2(3)=dotproduct(ninjaE3,spvae2e1)
      acd2(4)=abb2(7)
      acd2(5)=dotproduct(ninjaE3,spvae1e2)
      acd2(6)=abb2(8)
      acd2(7)=dotproduct(ninjaE3,spval5l4)
      acd2(8)=abb2(9)
      acd2(9)=dotproduct(ninjaE3,spval3l4)
      acd2(10)=abb2(11)
      acd2(11)=dotproduct(ninjaE3,spvak2l3)
      acd2(12)=abb2(12)
      acd2(13)=dotproduct(k2,ninjaA)
      acd2(14)=dotproduct(ninjaA,spvae2e1)
      acd2(15)=dotproduct(ninjaA,spvae1e2)
      acd2(16)=dotproduct(ninjaA,spval5l4)
      acd2(17)=dotproduct(ninjaA,spval3l4)
      acd2(18)=dotproduct(ninjaA,spvak2l3)
      acd2(19)=abb2(10)
      acd2(20)=acd2(1)*acd2(2)
      acd2(21)=acd2(3)*acd2(4)
      acd2(22)=acd2(5)*acd2(6)
      acd2(23)=acd2(7)*acd2(8)
      acd2(24)=acd2(9)*acd2(10)
      acd2(25)=acd2(11)*acd2(12)
      acd2(20)=acd2(25)+acd2(24)+acd2(23)+acd2(22)+acd2(20)+acd2(21)
      acd2(21)=acd2(13)*acd2(2)
      acd2(22)=acd2(14)*acd2(4)
      acd2(23)=acd2(15)*acd2(6)
      acd2(24)=acd2(16)*acd2(8)
      acd2(25)=acd2(17)*acd2(10)
      acd2(26)=acd2(18)*acd2(12)
      acd2(21)=acd2(19)+acd2(26)+acd2(25)+acd2(24)+acd2(23)+acd2(21)+acd2(22)
      brack(ninjaidxt1mu0)=acd2(20)
      brack(ninjaidxt0mu0)=acd2(21)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d2h4_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd2h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-2))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d2h4l131_qp
