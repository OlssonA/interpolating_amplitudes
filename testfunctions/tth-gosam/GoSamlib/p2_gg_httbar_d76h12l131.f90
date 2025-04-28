module     p2_gg_httbar_d76h12l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d76h12l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd76
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h12
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(75) :: acd76
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd76(1)=dotproduct(ninjaE3,spvae1l5)
      acd76(2)=abb76(28)
      acd76(3)=dotproduct(ninjaE3,spvae1k2)
      acd76(4)=abb76(17)
      acd76(5)=dotproduct(ninjaE3,spvak2e1)
      acd76(6)=abb76(13)
      acd76(7)=dotproduct(ninjaE3,spvae2e1)
      acd76(8)=abb76(22)
      acd76(9)=dotproduct(ninjaE3,spvae1e2)
      acd76(10)=abb76(30)
      acd76(11)=dotproduct(ninjaE3,spvae1l4)
      acd76(12)=abb76(33)
      acd76(13)=dotproduct(ninjaE3,spval4e1)
      acd76(14)=abb76(39)
      acd76(15)=dotproduct(ninjaA,ninjaE3)
      acd76(16)=dotproduct(ninjaE3,spval3e1)
      acd76(17)=abb76(10)
      acd76(18)=abb76(21)
      acd76(19)=abb76(43)
      acd76(20)=abb76(36)
      acd76(21)=abb76(45)
      acd76(22)=abb76(29)
      acd76(23)=abb76(42)
      acd76(24)=abb76(12)
      acd76(25)=dotproduct(ninjaE3,spvae1l3)
      acd76(26)=abb76(38)
      acd76(27)=abb76(23)
      acd76(28)=abb76(25)
      acd76(29)=abb76(16)
      acd76(30)=abb76(34)
      acd76(31)=dotproduct(ninjaA,ninjaA)
      acd76(32)=dotproduct(ninjaA,spvae1l5)
      acd76(33)=dotproduct(ninjaA,spvae1k2)
      acd76(34)=dotproduct(ninjaA,spvak2e1)
      acd76(35)=dotproduct(ninjaA,spvae2e1)
      acd76(36)=dotproduct(ninjaA,spvae1e2)
      acd76(37)=dotproduct(ninjaA,spvae1l4)
      acd76(38)=dotproduct(ninjaA,spval4e1)
      acd76(39)=abb76(15)
      acd76(40)=dotproduct(ninjaA,spval3e1)
      acd76(41)=dotproduct(ninjaA,spvae1l3)
      acd76(42)=abb76(20)
      acd76(43)=abb76(24)
      acd76(44)=abb76(11)
      acd76(45)=abb76(18)
      acd76(46)=abb76(31)
      acd76(47)=abb76(19)
      acd76(48)=abb76(26)
      acd76(49)=abb76(27)
      acd76(50)=abb76(37)
      acd76(51)=acd76(2)*acd76(1)
      acd76(52)=acd76(4)*acd76(3)
      acd76(53)=acd76(8)*acd76(7)
      acd76(54)=acd76(10)*acd76(9)
      acd76(55)=acd76(12)*acd76(11)
      acd76(56)=acd76(14)*acd76(13)
      acd76(51)=acd76(51)+acd76(52)+acd76(53)+acd76(54)+acd76(55)+acd76(56)
      acd76(52)=acd76(6)*acd76(5)
      acd76(52)=acd76(52)+acd76(51)
      acd76(53)=acd76(15)*acd76(52)
      acd76(54)=acd76(24)*acd76(3)
      acd76(55)=acd76(26)*acd76(25)
      acd76(56)=acd76(27)*acd76(9)
      acd76(57)=acd76(28)*acd76(11)
      acd76(54)=-acd76(54)+acd76(55)+acd76(56)-acd76(57)
      acd76(55)=-acd76(21)*acd76(1)
      acd76(55)=acd76(55)-acd76(54)
      acd76(55)=acd76(5)*acd76(55)
      acd76(56)=acd76(18)*acd76(3)
      acd76(57)=acd76(19)*acd76(9)
      acd76(58)=acd76(20)*acd76(11)
      acd76(56)=acd76(58)+acd76(56)+acd76(57)
      acd76(57)=acd76(17)*acd76(1)
      acd76(57)=acd76(57)+acd76(56)
      acd76(57)=acd76(16)*acd76(57)
      acd76(58)=acd76(22)*acd76(1)
      acd76(59)=acd76(29)*acd76(25)
      acd76(58)=acd76(58)+acd76(59)
      acd76(59)=acd76(7)*acd76(58)
      acd76(60)=acd76(23)*acd76(1)
      acd76(61)=acd76(30)*acd76(25)
      acd76(60)=acd76(60)+acd76(61)
      acd76(61)=acd76(13)*acd76(60)
      acd76(53)=2.0_ki*acd76(53)+acd76(55)+acd76(57)+acd76(61)+acd76(59)
      acd76(55)=ninjaP+acd76(31)
      acd76(51)=acd76(55)*acd76(51)
      acd76(54)=-acd76(34)*acd76(54)
      acd76(57)=2.0_ki*acd76(15)
      acd76(59)=acd76(2)*acd76(57)
      acd76(61)=acd76(22)*acd76(7)
      acd76(62)=acd76(23)*acd76(13)
      acd76(59)=acd76(62)+acd76(61)+acd76(59)
      acd76(59)=acd76(32)*acd76(59)
      acd76(56)=acd76(40)*acd76(56)
      acd76(61)=acd76(4)*acd76(57)
      acd76(62)=acd76(18)*acd76(16)
      acd76(63)=acd76(24)*acd76(5)
      acd76(61)=acd76(63)+acd76(61)+acd76(62)
      acd76(61)=acd76(33)*acd76(61)
      acd76(62)=acd76(27)*acd76(5)
      acd76(63)=acd76(10)*acd76(57)
      acd76(64)=acd76(19)*acd76(16)
      acd76(62)=-acd76(62)+acd76(63)+acd76(64)
      acd76(62)=acd76(36)*acd76(62)
      acd76(63)=acd76(28)*acd76(5)
      acd76(64)=acd76(12)*acd76(57)
      acd76(65)=acd76(20)*acd76(16)
      acd76(63)=acd76(63)+acd76(64)+acd76(65)
      acd76(63)=acd76(37)*acd76(63)
      acd76(64)=acd76(29)*acd76(7)
      acd76(65)=acd76(30)*acd76(13)
      acd76(66)=-acd76(26)*acd76(5)
      acd76(64)=acd76(65)+acd76(66)+acd76(64)
      acd76(64)=acd76(41)*acd76(64)
      acd76(55)=acd76(5)*acd76(55)
      acd76(65)=acd76(34)*acd76(57)
      acd76(55)=acd76(65)+acd76(55)
      acd76(55)=acd76(6)*acd76(55)
      acd76(65)=acd76(32)*acd76(16)
      acd76(66)=acd76(40)*acd76(1)
      acd76(65)=acd76(65)+acd76(66)
      acd76(65)=acd76(17)*acd76(65)
      acd76(66)=-acd76(34)*acd76(1)
      acd76(67)=-acd76(32)*acd76(5)
      acd76(66)=acd76(66)+acd76(67)
      acd76(66)=acd76(21)*acd76(66)
      acd76(67)=acd76(8)*acd76(57)
      acd76(58)=acd76(67)+acd76(58)
      acd76(58)=acd76(35)*acd76(58)
      acd76(67)=acd76(14)*acd76(57)
      acd76(60)=acd76(67)+acd76(60)
      acd76(60)=acd76(38)*acd76(60)
      acd76(57)=acd76(39)*acd76(57)
      acd76(67)=acd76(42)*acd76(16)
      acd76(68)=acd76(43)*acd76(1)
      acd76(69)=acd76(44)*acd76(3)
      acd76(70)=acd76(45)*acd76(5)
      acd76(71)=acd76(46)*acd76(25)
      acd76(72)=acd76(47)*acd76(7)
      acd76(73)=acd76(48)*acd76(9)
      acd76(74)=acd76(49)*acd76(11)
      acd76(75)=acd76(50)*acd76(13)
      acd76(51)=acd76(75)+acd76(74)+acd76(73)+acd76(72)+acd76(71)+acd76(70)+acd&
      &76(69)+acd76(68)+acd76(67)+acd76(57)+acd76(64)+acd76(60)+acd76(63)+acd76&
      &(62)+acd76(58)+acd76(61)+acd76(66)+acd76(65)+acd76(55)+acd76(54)+acd76(5&
      &6)+acd76(59)+acd76(51)
      brack(ninjaidxt1mu0)=acd76(53)
      brack(ninjaidxt0mu0)=acd76(51)
      brack(ninjaidxt0mu2)=acd76(52)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d76h12_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd76h12
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k3+k5
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
end module     p2_gg_httbar_d76h12l131
