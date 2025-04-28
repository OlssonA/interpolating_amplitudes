module     p0_ubaru_httbar_d43h14l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity14d43h14l131.f90
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
      use p0_ubaru_httbar_abbrevd43h14
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(31) :: acd43
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd43(1)=dotproduct(ninjaA,ninjaE3)
      acd43(2)=abb43(21)
      acd43(3)=dotproduct(ninjaA,spvak2k1)
      acd43(4)=dotproduct(ninjaE3,spvak2l3)
      acd43(5)=abb43(11)
      acd43(6)=dotproduct(ninjaE3,spvak2l4)
      acd43(7)=abb43(12)
      acd43(8)=dotproduct(ninjaE3,spvak2l5)
      acd43(9)=abb43(13)
      acd43(10)=dotproduct(ninjaE3,spval3l5)
      acd43(11)=abb43(16)
      acd43(12)=dotproduct(ninjaE3,spval3l4)
      acd43(13)=abb43(18)
      acd43(14)=dotproduct(ninjaA,spvak2l3)
      acd43(15)=dotproduct(ninjaE3,spvak2k1)
      acd43(16)=dotproduct(ninjaA,spvak2l4)
      acd43(17)=dotproduct(ninjaA,spvak2l5)
      acd43(18)=dotproduct(ninjaA,spval3l5)
      acd43(19)=dotproduct(ninjaA,spval3l4)
      acd43(20)=abb43(14)
      acd43(21)=abb43(23)
      acd43(22)=abb43(24)
      acd43(23)=dotproduct(ninjaE3,spval3k1)
      acd43(24)=abb43(17)
      acd43(25)=acd43(14)*acd43(5)
      acd43(26)=acd43(16)*acd43(7)
      acd43(27)=acd43(17)*acd43(9)
      acd43(28)=acd43(18)*acd43(11)
      acd43(29)=acd43(19)*acd43(13)
      acd43(25)=acd43(20)+acd43(29)+acd43(28)+acd43(27)+acd43(26)+acd43(25)
      acd43(25)=acd43(15)*acd43(25)
      acd43(26)=acd43(7)*acd43(6)
      acd43(27)=acd43(9)*acd43(8)
      acd43(28)=acd43(4)*acd43(5)
      acd43(29)=acd43(10)*acd43(11)
      acd43(30)=acd43(12)*acd43(13)
      acd43(26)=acd43(30)+acd43(26)+acd43(27)+acd43(28)+acd43(29)
      acd43(27)=acd43(3)*acd43(26)
      acd43(28)=acd43(2)*acd43(1)
      acd43(29)=acd43(21)*acd43(6)
      acd43(30)=acd43(22)*acd43(8)
      acd43(31)=acd43(24)*acd43(23)
      acd43(25)=acd43(31)+acd43(30)+acd43(29)-2.0_ki*acd43(28)+acd43(27)+acd43(&
      &25)
      acd43(26)=acd43(15)*acd43(26)
      brack(ninjaidxt3mu0)=acd43(26)
      brack(ninjaidxt2mu0)=acd43(25)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d43h14_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd43h14
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d43h14l131
