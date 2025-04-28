module     p2_gg_httbar_d172h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d172h0l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd172h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(23) :: acd172
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd172(1)=dotproduct(ninjaE3,spvak2e1)
      acd172(2)=dotproduct(ninjaE3,spvae1k2)
      acd172(3)=abb172(15)
      acd172(4)=dotproduct(ninjaE3,spval4e1)
      acd172(5)=abb172(26)
      acd172(6)=dotproduct(ninjaE3,spvae2e1)
      acd172(7)=abb172(29)
      acd172(8)=dotproduct(ninjaE3,spval5e1)
      acd172(9)=abb172(30)
      acd172(10)=dotproduct(ninjaE3,spvae1e2)
      acd172(11)=abb172(23)
      acd172(12)=dotproduct(ninjaE3,spvae1l5)
      acd172(13)=abb172(24)
      acd172(14)=dotproduct(ninjaE3,spvae1l4)
      acd172(15)=abb172(66)
      acd172(16)=abb172(54)
      acd172(17)=abb172(39)
      acd172(18)=abb172(63)
      acd172(19)=acd172(1)*acd172(3)
      acd172(20)=acd172(6)*acd172(7)
      acd172(21)=acd172(8)*acd172(9)
      acd172(22)=acd172(4)*acd172(5)
      acd172(19)=acd172(22)+acd172(21)+acd172(19)+acd172(20)
      acd172(19)=acd172(2)*acd172(19)
      acd172(20)=acd172(14)*acd172(18)
      acd172(21)=-acd172(12)*acd172(15)
      acd172(22)=acd172(10)*acd172(17)
      acd172(20)=acd172(22)+acd172(20)+acd172(21)
      acd172(20)=acd172(8)*acd172(20)
      acd172(21)=-acd172(14)*acd172(15)
      acd172(22)=acd172(12)*acd172(13)
      acd172(23)=acd172(10)*acd172(11)
      acd172(21)=acd172(23)+acd172(21)+acd172(22)
      acd172(21)=acd172(4)*acd172(21)
      acd172(22)=acd172(10)*acd172(6)*acd172(16)
      acd172(19)=acd172(19)+acd172(21)+acd172(22)+acd172(20)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd172(19)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd172h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(72) :: acd172
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd172(1)=dotproduct(ninjaE3,spvak2e1)
      acd172(2)=dotproduct(ninjaE4,spvae1k2)
      acd172(3)=abb172(15)
      acd172(4)=dotproduct(ninjaE3,spvae1k2)
      acd172(5)=dotproduct(ninjaE4,spvak2e1)
      acd172(6)=dotproduct(ninjaE4,spvae2e1)
      acd172(7)=abb172(29)
      acd172(8)=dotproduct(ninjaE4,spval4e1)
      acd172(9)=abb172(26)
      acd172(10)=dotproduct(ninjaE4,spval5e1)
      acd172(11)=abb172(30)
      acd172(12)=dotproduct(ninjaE3,spvae2e1)
      acd172(13)=dotproduct(ninjaE4,spvae1e2)
      acd172(14)=abb172(54)
      acd172(15)=dotproduct(ninjaE3,spval4e1)
      acd172(16)=abb172(23)
      acd172(17)=dotproduct(ninjaE4,spvae1l5)
      acd172(18)=abb172(24)
      acd172(19)=dotproduct(ninjaE4,spvae1l4)
      acd172(20)=abb172(66)
      acd172(21)=dotproduct(ninjaE3,spvae1e2)
      acd172(22)=abb172(39)
      acd172(23)=dotproduct(ninjaE3,spvae1l5)
      acd172(24)=dotproduct(ninjaE3,spval5e1)
      acd172(25)=abb172(63)
      acd172(26)=dotproduct(ninjaE3,spvae1l4)
      acd172(27)=abb172(13)
      acd172(28)=dotproduct(ninjaA,ninjaE3)
      acd172(29)=dotproduct(ninjaA,spvak2e1)
      acd172(30)=dotproduct(ninjaA,spvae1k2)
      acd172(31)=dotproduct(ninjaA,spvae2e1)
      acd172(32)=dotproduct(ninjaA,spval4e1)
      acd172(33)=dotproduct(ninjaA,spvae1e2)
      acd172(34)=dotproduct(ninjaA,spvae1l5)
      acd172(35)=dotproduct(ninjaA,spval5e1)
      acd172(36)=dotproduct(ninjaA,spvae1l4)
      acd172(37)=abb172(14)
      acd172(38)=abb172(27)
      acd172(39)=abb172(17)
      acd172(40)=dotproduct(ninjaE3,spval3e1)
      acd172(41)=abb172(21)
      acd172(42)=abb172(22)
      acd172(43)=abb172(48)
      acd172(44)=abb172(56)
      acd172(45)=dotproduct(ninjaE3,spvae1l3)
      acd172(46)=abb172(25)
      acd172(47)=abb172(38)
      acd172(48)=abb172(57)
      acd172(49)=dotproduct(ninjaA,ninjaA)
      acd172(50)=dotproduct(ninjaA,spval3e1)
      acd172(51)=dotproduct(ninjaA,spvae1l3)
      acd172(52)=abb172(12)
      acd172(53)=acd172(7)*acd172(6)
      acd172(54)=acd172(3)*acd172(5)
      acd172(55)=acd172(10)*acd172(11)
      acd172(56)=acd172(8)*acd172(9)
      acd172(53)=acd172(54)+acd172(53)+acd172(55)+acd172(56)
      acd172(53)=acd172(53)*acd172(4)
      acd172(54)=acd172(10)*acd172(23)
      acd172(55)=acd172(8)*acd172(26)
      acd172(56)=acd172(24)*acd172(17)
      acd172(57)=acd172(15)*acd172(19)
      acd172(54)=acd172(54)+acd172(55)+acd172(56)+acd172(57)
      acd172(54)=acd172(54)*acd172(20)
      acd172(55)=acd172(18)*acd172(17)
      acd172(56)=acd172(16)*acd172(13)
      acd172(57)=acd172(2)*acd172(9)
      acd172(55)=acd172(57)+acd172(55)+acd172(56)
      acd172(55)=acd172(55)*acd172(15)
      acd172(56)=acd172(25)*acd172(19)
      acd172(57)=acd172(22)*acd172(13)
      acd172(58)=acd172(2)*acd172(11)
      acd172(56)=acd172(58)+acd172(56)+acd172(57)
      acd172(56)=acd172(56)*acd172(24)
      acd172(57)=acd172(14)*acd172(6)
      acd172(58)=acd172(10)*acd172(22)
      acd172(59)=acd172(8)*acd172(16)
      acd172(57)=acd172(57)+acd172(58)+acd172(59)
      acd172(57)=acd172(57)*acd172(21)
      acd172(58)=acd172(7)*acd172(12)
      acd172(59)=acd172(3)*acd172(1)
      acd172(58)=acd172(58)+acd172(59)
      acd172(59)=acd172(58)*acd172(2)
      acd172(60)=acd172(13)*acd172(12)*acd172(14)
      acd172(61)=acd172(18)*acd172(23)
      acd172(62)=acd172(61)*acd172(8)
      acd172(63)=acd172(25)*acd172(26)
      acd172(64)=acd172(63)*acd172(10)
      acd172(53)=acd172(53)-acd172(54)+acd172(60)+acd172(59)+acd172(62)+acd172(&
      &64)+acd172(57)+acd172(27)+acd172(55)+acd172(56)
      acd172(54)=acd172(7)*acd172(31)
      acd172(55)=acd172(3)*acd172(29)
      acd172(56)=acd172(35)*acd172(11)
      acd172(57)=acd172(32)*acd172(9)
      acd172(54)=acd172(54)+acd172(55)+acd172(56)+acd172(57)+acd172(38)
      acd172(55)=acd172(4)*acd172(54)
      acd172(56)=-acd172(35)*acd172(23)
      acd172(57)=-acd172(32)*acd172(26)
      acd172(59)=-acd172(24)*acd172(34)
      acd172(60)=-acd172(15)*acd172(36)
      acd172(56)=acd172(60)+acd172(59)+acd172(56)+acd172(57)
      acd172(56)=acd172(20)*acd172(56)
      acd172(57)=acd172(14)*acd172(31)
      acd172(57)=acd172(57)+acd172(43)
      acd172(59)=acd172(35)*acd172(22)
      acd172(60)=acd172(32)*acd172(16)
      acd172(59)=acd172(60)+acd172(59)+acd172(57)
      acd172(59)=acd172(21)*acd172(59)
      acd172(58)=acd172(30)*acd172(58)
      acd172(60)=acd172(25)*acd172(36)
      acd172(62)=acd172(22)*acd172(33)
      acd172(60)=acd172(47)+acd172(60)+acd172(62)
      acd172(62)=acd172(30)*acd172(11)
      acd172(62)=acd172(62)+acd172(60)
      acd172(62)=acd172(24)*acd172(62)
      acd172(64)=acd172(18)*acd172(34)
      acd172(65)=acd172(16)*acd172(33)
      acd172(64)=acd172(42)+acd172(64)+acd172(65)
      acd172(65)=acd172(30)*acd172(9)
      acd172(65)=acd172(65)+acd172(64)
      acd172(65)=acd172(15)*acd172(65)
      acd172(66)=acd172(46)*acd172(45)
      acd172(67)=acd172(41)*acd172(40)
      acd172(68)=acd172(27)*acd172(28)
      acd172(69)=acd172(1)*acd172(37)
      acd172(70)=acd172(26)*acd172(48)
      acd172(71)=acd172(23)*acd172(44)
      acd172(72)=acd172(14)*acd172(33)
      acd172(72)=acd172(39)+acd172(72)
      acd172(72)=acd172(12)*acd172(72)
      acd172(63)=acd172(35)*acd172(63)
      acd172(61)=acd172(32)*acd172(61)
      acd172(55)=acd172(56)+acd172(55)+acd172(65)+acd172(62)+acd172(59)+acd172(&
      &58)+acd172(61)+acd172(63)+acd172(72)+acd172(71)+acd172(70)+acd172(69)+2.&
      &0_ki*acd172(68)+acd172(66)+acd172(67)
      acd172(56)=ninjaP*acd172(53)
      acd172(54)=acd172(30)*acd172(54)
      acd172(58)=acd172(35)*acd172(60)
      acd172(59)=acd172(32)*acd172(64)
      acd172(60)=-acd172(35)*acd172(34)
      acd172(61)=-acd172(32)*acd172(36)
      acd172(60)=acd172(60)+acd172(61)
      acd172(60)=acd172(20)*acd172(60)
      acd172(57)=acd172(33)*acd172(57)
      acd172(61)=acd172(46)*acd172(51)
      acd172(62)=acd172(41)*acd172(50)
      acd172(63)=acd172(29)*acd172(37)
      acd172(64)=acd172(27)*acd172(49)
      acd172(65)=acd172(36)*acd172(48)
      acd172(66)=acd172(34)*acd172(44)
      acd172(67)=acd172(31)*acd172(39)
      acd172(54)=acd172(56)+acd172(60)+acd172(54)+acd172(59)+acd172(58)+acd172(&
      &67)+acd172(66)+acd172(65)+acd172(64)+acd172(63)+acd172(62)+acd172(52)+ac&
      &d172(61)+acd172(57)
      brack(ninjaidxt1mu0)=acd172(55)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd172(54)
      brack(ninjaidxt0mu2)=acd172(53)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d172h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd172h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d172h0l131
