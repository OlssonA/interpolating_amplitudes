module     p0_ubaru_httbar_d21h13l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d21h13l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd21h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(30) :: acd21
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd21(1)=dotproduct(k2,ninjaE3)
      acd21(2)=abb21(12)
      acd21(3)=dotproduct(l3,ninjaE3)
      acd21(4)=abb21(15)
      acd21(5)=dotproduct(l4,ninjaE3)
      acd21(6)=abb21(19)
      acd21(7)=dotproduct(ninjaE3,spval4l3)
      acd21(8)=abb21(13)
      acd21(9)=dotproduct(ninjaE3,spval3k2)
      acd21(10)=abb21(14)
      acd21(11)=dotproduct(ninjaE3,spval3l4)
      acd21(12)=abb21(18)
      acd21(13)=dotproduct(ninjaE3,spvak2l3)
      acd21(14)=abb21(20)
      acd21(15)=dotproduct(ninjaE3,spvak2l4)
      acd21(16)=abb21(21)
      acd21(17)=dotproduct(ninjaE3,spvak1l3)
      acd21(18)=abb21(22)
      acd21(19)=dotproduct(ninjaE3,spvak1l4)
      acd21(20)=abb21(23)
      acd21(21)=acd21(2)*acd21(1)
      acd21(22)=acd21(4)*acd21(3)
      acd21(23)=acd21(6)*acd21(5)
      acd21(24)=acd21(8)*acd21(7)
      acd21(25)=acd21(10)*acd21(9)
      acd21(26)=acd21(12)*acd21(11)
      acd21(27)=acd21(14)*acd21(13)
      acd21(28)=acd21(16)*acd21(15)
      acd21(29)=acd21(18)*acd21(17)
      acd21(30)=acd21(20)*acd21(19)
      acd21(21)=acd21(30)+acd21(29)+acd21(28)+acd21(27)+acd21(26)+acd21(25)+acd&
      &21(24)+acd21(23)+acd21(21)+acd21(22)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd21(21)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d21h13_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd21h13_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k4+k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d21h13l131_qp
