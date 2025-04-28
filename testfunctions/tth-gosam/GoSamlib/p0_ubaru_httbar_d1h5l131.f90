module     p0_ubaru_httbar_d1h5l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d1h5l131.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd1h5
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(13) :: acd1
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd1(1)=dotproduct(k2,ninjaE3)
      acd1(2)=dotproduct(ninjaE3,spvak1k2)
      acd1(3)=abb1(9)
      acd1(4)=dotproduct(ninjaE3,spval3k2)
      acd1(5)=abb1(11)
      acd1(6)=dotproduct(ninjaE3,spvak1l4)
      acd1(7)=dotproduct(ninjaE3,spval5k2)
      acd1(8)=abb1(15)
      acd1(9)=dotproduct(ninjaE3,spvak1l3)
      acd1(10)=abb1(17)
      acd1(11)=acd1(3)*acd1(1)
      acd1(12)=acd1(5)*acd1(4)
      acd1(11)=acd1(11)+acd1(12)
      acd1(11)=acd1(2)*acd1(11)
      acd1(12)=-acd1(8)*acd1(6)
      acd1(13)=acd1(10)*acd1(9)
      acd1(12)=acd1(13)+acd1(12)
      acd1(12)=acd1(7)*acd1(12)
      acd1(11)=acd1(12)+acd1(11)
      brack(ninjaidxt2mu0)=acd1(11)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd1h5
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(40) :: acd1
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd1(1)=dotproduct(k2,ninjaE3)
      acd1(2)=dotproduct(ninjaE4,spvak1k2)
      acd1(3)=abb1(9)
      acd1(4)=dotproduct(k2,ninjaE4)
      acd1(5)=dotproduct(ninjaE3,spvak1k2)
      acd1(6)=dotproduct(ninjaE4,spval3k2)
      acd1(7)=abb1(11)
      acd1(8)=dotproduct(ninjaE3,spval3k2)
      acd1(9)=dotproduct(ninjaE3,spval5k2)
      acd1(10)=dotproduct(ninjaE4,spvak1l4)
      acd1(11)=abb1(15)
      acd1(12)=dotproduct(ninjaE4,spvak1l3)
      acd1(13)=abb1(17)
      acd1(14)=dotproduct(ninjaE3,spvak1l4)
      acd1(15)=dotproduct(ninjaE4,spval5k2)
      acd1(16)=dotproduct(ninjaE3,spvak1l3)
      acd1(17)=dotproduct(k2,ninjaA)
      acd1(18)=dotproduct(ninjaA,spvak1k2)
      acd1(19)=abb1(19)
      acd1(20)=dotproduct(ninjaA,spval3k2)
      acd1(21)=dotproduct(ninjaA,spval5k2)
      acd1(22)=dotproduct(ninjaA,spvak1l4)
      acd1(23)=dotproduct(ninjaA,spvak1l3)
      acd1(24)=abb1(10)
      acd1(25)=abb1(23)
      acd1(26)=abb1(13)
      acd1(27)=abb1(27)
      acd1(28)=abb1(26)
      acd1(29)=abb1(22)
      acd1(30)=acd1(1)*acd1(2)
      acd1(31)=acd1(5)*acd1(4)
      acd1(30)=acd1(30)+acd1(31)
      acd1(30)=acd1(30)*acd1(3)
      acd1(31)=acd1(13)*acd1(12)
      acd1(32)=acd1(11)*acd1(10)
      acd1(31)=acd1(31)-acd1(32)
      acd1(31)=acd1(31)*acd1(9)
      acd1(32)=acd1(13)*acd1(16)
      acd1(33)=acd1(11)*acd1(14)
      acd1(32)=acd1(32)-acd1(33)
      acd1(33)=acd1(32)*acd1(15)
      acd1(34)=acd1(7)*acd1(8)
      acd1(35)=acd1(34)*acd1(2)
      acd1(36)=acd1(6)*acd1(5)*acd1(7)
      acd1(30)=acd1(31)+acd1(30)+acd1(33)+acd1(35)+acd1(36)
      acd1(31)=acd1(21)*acd1(32)
      acd1(32)=acd1(13)*acd1(23)
      acd1(33)=acd1(11)*acd1(22)
      acd1(32)=acd1(26)+acd1(32)-acd1(33)
      acd1(33)=acd1(9)*acd1(32)
      acd1(35)=acd1(18)*acd1(1)
      acd1(36)=acd1(5)*acd1(17)
      acd1(35)=acd1(35)+acd1(36)
      acd1(35)=acd1(3)*acd1(35)
      acd1(36)=acd1(16)*acd1(28)
      acd1(37)=acd1(14)*acd1(27)
      acd1(38)=acd1(8)*acd1(25)
      acd1(39)=acd1(1)*acd1(19)
      acd1(34)=acd1(18)*acd1(34)
      acd1(40)=acd1(7)*acd1(20)
      acd1(40)=acd1(24)+acd1(40)
      acd1(40)=acd1(5)*acd1(40)
      acd1(31)=acd1(35)+acd1(40)+acd1(34)+acd1(33)+acd1(39)+acd1(38)+acd1(36)+a&
      &cd1(37)+acd1(31)
      acd1(33)=ninjaP*acd1(30)
      acd1(32)=acd1(21)*acd1(32)
      acd1(34)=acd1(7)*acd1(18)
      acd1(34)=acd1(34)+acd1(25)
      acd1(34)=acd1(20)*acd1(34)
      acd1(35)=acd1(3)*acd1(17)
      acd1(35)=acd1(35)+acd1(24)
      acd1(35)=acd1(18)*acd1(35)
      acd1(36)=acd1(23)*acd1(28)
      acd1(37)=acd1(22)*acd1(27)
      acd1(38)=acd1(17)*acd1(19)
      acd1(32)=acd1(33)+acd1(38)+acd1(37)+acd1(29)+acd1(36)+acd1(32)+acd1(35)+a&
      &cd1(34)
      brack(ninjaidxt1mu0)=acd1(31)
      brack(ninjaidxt0mu0)=acd1(32)
      brack(ninjaidxt0mu2)=acd1(30)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d1h5_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd1h5
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k5
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d1h5l131
