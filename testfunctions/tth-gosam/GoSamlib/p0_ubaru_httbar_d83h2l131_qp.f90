module     p0_ubaru_httbar_d83h2l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d83h2l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt1mu0 = 0
   integer, parameter :: ninjaidxt0mu0 = 1
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd83h2_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd83
      complex(ki), dimension (0:*), intent(inout) :: brack
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd83h2_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(18) :: acd83
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd83(1)=dotproduct(k2,ninjaE3)
      acd83(2)=abb83(6)
      acd83(3)=dotproduct(ninjaA,ninjaE3)
      acd83(4)=abb83(13)
      acd83(5)=dotproduct(ninjaE3,spval4k1)
      acd83(6)=abb83(22)
      acd83(7)=dotproduct(ninjaE3,spvak2k1)
      acd83(8)=dotproduct(ninjaE3,spval4k2)
      acd83(9)=abb83(14)
      acd83(10)=dotproduct(ninjaE3,spvak1k2)
      acd83(11)=abb83(23)
      acd83(12)=dotproduct(ninjaE3,spvak2l4)
      acd83(13)=dotproduct(ninjaE3,spval4l5)
      acd83(14)=dotproduct(ninjaE3,spval5k1)
      acd83(15)=-acd83(13)*acd83(14)
      acd83(16)=2.0_ki*acd83(3)
      acd83(17)=acd83(5)*acd83(16)
      acd83(15)=acd83(17)+acd83(15)
      acd83(15)=acd83(6)*acd83(15)
      acd83(17)=-acd83(11)*acd83(10)
      acd83(18)=acd83(8)*acd83(9)
      acd83(17)=acd83(17)+acd83(18)
      acd83(17)=acd83(7)*acd83(17)
      acd83(16)=acd83(4)*acd83(16)
      acd83(18)=acd83(1)*acd83(2)
      acd83(16)=acd83(16)+acd83(18)
      acd83(16)=acd83(1)*acd83(16)
      acd83(18)=acd83(8)*acd83(11)*acd83(12)
      acd83(15)=acd83(16)+acd83(18)+acd83(17)+acd83(15)
      brack(ninjaidxt1mu0)=0.0_ki
      brack(ninjaidxt0mu0)=acd83(15)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d83h2_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd83h2_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2
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
end module     p0_ubaru_httbar_d83h2l131_qp
