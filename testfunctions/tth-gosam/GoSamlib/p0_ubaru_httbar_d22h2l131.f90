module     p0_ubaru_httbar_d22h2l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d22h2l131.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
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
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd22h2
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(30) :: acd22
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd22(1)=dotproduct(k2,ninjaE3)
      acd22(2)=abb22(13)
      acd22(3)=dotproduct(l3,ninjaE3)
      acd22(4)=abb22(17)
      acd22(5)=dotproduct(l5,ninjaE3)
      acd22(6)=abb22(35)
      acd22(7)=dotproduct(ninjaE3,spval3k2)
      acd22(8)=abb22(12)
      acd22(9)=dotproduct(ninjaE3,spval5k2)
      acd22(10)=abb22(14)
      acd22(11)=dotproduct(ninjaE3,spval5k1)
      acd22(12)=abb22(16)
      acd22(13)=dotproduct(ninjaE3,spval5l3)
      acd22(14)=abb22(18)
      acd22(15)=dotproduct(ninjaE3,spval3k1)
      acd22(16)=abb22(19)
      acd22(17)=dotproduct(ninjaE3,spvak2l3)
      acd22(18)=abb22(20)
      acd22(19)=dotproduct(ninjaE3,spval3l5)
      acd22(20)=abb22(21)
      acd22(21)=acd22(2)*acd22(1)
      acd22(22)=acd22(4)*acd22(3)
      acd22(23)=acd22(6)*acd22(5)
      acd22(24)=acd22(8)*acd22(7)
      acd22(25)=acd22(10)*acd22(9)
      acd22(26)=acd22(12)*acd22(11)
      acd22(27)=acd22(14)*acd22(13)
      acd22(28)=acd22(16)*acd22(15)
      acd22(29)=acd22(18)*acd22(17)
      acd22(30)=acd22(20)*acd22(19)
      acd22(21)=acd22(30)+acd22(29)+acd22(28)+acd22(27)+acd22(26)+acd22(25)+acd&
      &22(24)+acd22(23)+acd22(21)+acd22(22)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd22(21)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d22h2_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd22h2
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = - a(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d22h2l131
