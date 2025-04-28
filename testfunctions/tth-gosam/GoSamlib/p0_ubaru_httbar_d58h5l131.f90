module     p0_ubaru_httbar_d58h5l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d58h5l131.f90
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
      use p0_ubaru_httbar_abbrevd58h5
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd58
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd58h5
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(49) :: acd58
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd58(1)=dotproduct(k2,ninjaE3)
      acd58(2)=abb58(34)
      acd58(3)=dotproduct(ninjaE3,spval5k2)
      acd58(4)=abb58(16)
      acd58(5)=dotproduct(ninjaE3,spvak1k2)
      acd58(6)=abb58(21)
      acd58(7)=dotproduct(ninjaA,ninjaE3)
      acd58(8)=abb58(19)
      acd58(9)=dotproduct(ninjaE3,spvak1l4)
      acd58(10)=abb58(15)
      acd58(11)=dotproduct(k2,ninjaA)
      acd58(12)=dotproduct(ninjaA,ninjaA)
      acd58(13)=dotproduct(ninjaA,spvak1k2)
      acd58(14)=abb58(18)
      acd58(15)=dotproduct(ninjaA,spval5k2)
      acd58(16)=abb58(32)
      acd58(17)=dotproduct(ninjaA,spvak1l4)
      acd58(18)=abb58(10)
      acd58(19)=dotproduct(ninjaE3,spval3k2)
      acd58(20)=abb58(11)
      acd58(21)=dotproduct(ninjaE3,spval4l3)
      acd58(22)=abb58(12)
      acd58(23)=abb58(13)
      acd58(24)=abb58(14)
      acd58(25)=dotproduct(ninjaE3,spval5l3)
      acd58(26)=abb58(20)
      acd58(27)=dotproduct(ninjaE3,spval4k2)
      acd58(28)=abb58(23)
      acd58(29)=dotproduct(ninjaE3,spval3l5)
      acd58(30)=abb58(24)
      acd58(31)=dotproduct(ninjaE3,spval3l4)
      acd58(32)=abb58(35)
      acd58(33)=acd58(2)*acd58(1)
      acd58(34)=acd58(4)*acd58(3)
      acd58(35)=acd58(6)*acd58(5)
      acd58(33)=acd58(35)+acd58(33)+acd58(34)
      acd58(34)=acd58(7)*acd58(33)
      acd58(35)=acd58(8)*acd58(5)
      acd58(36)=acd58(1)*acd58(35)
      acd58(37)=acd58(10)*acd58(9)
      acd58(38)=acd58(3)*acd58(37)
      acd58(34)=acd58(38)+2.0_ki*acd58(34)+acd58(36)
      acd58(36)=acd58(12)+ninjaP
      acd58(36)=acd58(33)*acd58(36)
      acd58(38)=2.0_ki*acd58(7)
      acd58(39)=acd58(2)*acd58(38)
      acd58(35)=acd58(39)+acd58(35)
      acd58(35)=acd58(11)*acd58(35)
      acd58(39)=acd58(8)*acd58(1)
      acd58(40)=acd58(6)*acd58(38)
      acd58(39)=acd58(40)+acd58(39)
      acd58(39)=acd58(13)*acd58(39)
      acd58(40)=acd58(4)*acd58(38)
      acd58(37)=acd58(40)+acd58(37)
      acd58(37)=acd58(15)*acd58(37)
      acd58(40)=acd58(17)*acd58(10)
      acd58(40)=acd58(18)+acd58(40)
      acd58(40)=acd58(3)*acd58(40)
      acd58(41)=acd58(14)*acd58(1)
      acd58(38)=acd58(16)*acd58(38)
      acd58(42)=acd58(20)*acd58(19)
      acd58(43)=acd58(22)*acd58(21)
      acd58(44)=acd58(23)*acd58(5)
      acd58(45)=acd58(24)*acd58(9)
      acd58(46)=acd58(26)*acd58(25)
      acd58(47)=acd58(28)*acd58(27)
      acd58(48)=acd58(30)*acd58(29)
      acd58(49)=acd58(32)*acd58(31)
      acd58(35)=acd58(49)+acd58(48)+acd58(47)+acd58(46)+acd58(45)+acd58(44)+acd&
      &58(43)+acd58(42)+acd58(38)+acd58(41)+acd58(37)+acd58(39)+acd58(35)+acd58&
      &(36)+acd58(40)
      brack(ninjaidxt1mu0)=acd58(34)
      brack(ninjaidxt0mu0)=acd58(35)
      brack(ninjaidxt0mu2)=acd58(33)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d58h5_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd58h5
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k4
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d58h5l131
