module     p0_ubaru_httbar_d77h2l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d77h2l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h2_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(26) :: acd77
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd77(1)=dotproduct(k2,ninjaE3)
      acd77(2)=dotproduct(ninjaE3,spval4k1)
      acd77(3)=abb77(40)
      acd77(4)=dotproduct(ninjaE3,spval5k1)
      acd77(5)=abb77(26)
      acd77(6)=dotproduct(ninjaE3,spval3k1)
      acd77(7)=abb77(38)
      acd77(8)=dotproduct(ninjaE3,spvak2k1)
      acd77(9)=dotproduct(ninjaE3,spval3k2)
      acd77(10)=abb77(12)
      acd77(11)=dotproduct(ninjaE3,spval5l3)
      acd77(12)=abb77(13)
      acd77(13)=dotproduct(ninjaE3,spval4l3)
      acd77(14)=abb77(14)
      acd77(15)=dotproduct(ninjaE3,spval5k2)
      acd77(16)=abb77(20)
      acd77(17)=dotproduct(ninjaE3,spval4k2)
      acd77(18)=abb77(21)
      acd77(19)=dotproduct(ninjaE3,spvak2l3)
      acd77(20)=abb77(16)
      acd77(21)=abb77(44)
      acd77(22)=acd77(10)*acd77(9)
      acd77(23)=acd77(12)*acd77(11)
      acd77(24)=acd77(14)*acd77(13)
      acd77(25)=acd77(16)*acd77(15)
      acd77(26)=acd77(18)*acd77(17)
      acd77(22)=acd77(26)+acd77(25)+acd77(24)+acd77(23)+acd77(22)
      acd77(22)=acd77(8)*acd77(22)
      acd77(23)=acd77(3)*acd77(2)
      acd77(24)=acd77(5)*acd77(4)
      acd77(25)=acd77(7)*acd77(6)
      acd77(23)=acd77(25)+acd77(23)+acd77(24)
      acd77(23)=acd77(1)*acd77(23)
      acd77(24)=acd77(20)*acd77(2)
      acd77(25)=acd77(21)*acd77(4)
      acd77(24)=acd77(25)+acd77(24)
      acd77(24)=acd77(19)*acd77(24)
      acd77(22)=acd77(22)+acd77(23)+acd77(24)
      brack(ninjaidxt2mu0)=acd77(22)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h2_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(67) :: acd77
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd77(1)=dotproduct(k2,ninjaE3)
      acd77(2)=dotproduct(ninjaE4,spval4k1)
      acd77(3)=abb77(40)
      acd77(4)=dotproduct(ninjaE4,spval5k1)
      acd77(5)=abb77(26)
      acd77(6)=dotproduct(ninjaE4,spval3k1)
      acd77(7)=abb77(38)
      acd77(8)=dotproduct(k2,ninjaE4)
      acd77(9)=dotproduct(ninjaE3,spval4k1)
      acd77(10)=dotproduct(ninjaE3,spval5k1)
      acd77(11)=dotproduct(ninjaE3,spval3k1)
      acd77(12)=dotproduct(ninjaE3,spvak2k1)
      acd77(13)=dotproduct(ninjaE4,spval3k2)
      acd77(14)=abb77(12)
      acd77(15)=dotproduct(ninjaE4,spval5l3)
      acd77(16)=abb77(13)
      acd77(17)=dotproduct(ninjaE4,spval4l3)
      acd77(18)=abb77(14)
      acd77(19)=dotproduct(ninjaE4,spval5k2)
      acd77(20)=abb77(20)
      acd77(21)=dotproduct(ninjaE4,spval4k2)
      acd77(22)=abb77(21)
      acd77(23)=dotproduct(ninjaE3,spval3k2)
      acd77(24)=dotproduct(ninjaE4,spvak2k1)
      acd77(25)=dotproduct(ninjaE3,spval5l3)
      acd77(26)=dotproduct(ninjaE3,spval4l3)
      acd77(27)=dotproduct(ninjaE3,spvak2l3)
      acd77(28)=abb77(16)
      acd77(29)=abb77(44)
      acd77(30)=dotproduct(ninjaE4,spvak2l3)
      acd77(31)=dotproduct(ninjaE3,spval5k2)
      acd77(32)=dotproduct(ninjaE3,spval4k2)
      acd77(33)=abb77(11)
      acd77(34)=dotproduct(k2,ninjaA)
      acd77(35)=dotproduct(ninjaA,spval4k1)
      acd77(36)=dotproduct(ninjaA,spval5k1)
      acd77(37)=dotproduct(ninjaA,spval3k1)
      acd77(38)=abb77(15)
      acd77(39)=dotproduct(ninjaA,ninjaE3)
      acd77(40)=dotproduct(ninjaA,spvak2k1)
      acd77(41)=dotproduct(ninjaA,spval3k2)
      acd77(42)=dotproduct(ninjaA,spval5l3)
      acd77(43)=dotproduct(ninjaA,spval4l3)
      acd77(44)=dotproduct(ninjaA,spvak2l3)
      acd77(45)=dotproduct(ninjaA,spval5k2)
      acd77(46)=dotproduct(ninjaA,spval4k2)
      acd77(47)=abb77(10)
      acd77(48)=abb77(17)
      acd77(49)=abb77(30)
      acd77(50)=abb77(22)
      acd77(51)=abb77(36)
      acd77(52)=dotproduct(ninjaA,ninjaA)
      acd77(53)=abb77(19)
      acd77(54)=acd77(22)*acd77(21)
      acd77(55)=acd77(20)*acd77(19)
      acd77(56)=acd77(18)*acd77(17)
      acd77(57)=acd77(16)*acd77(15)
      acd77(58)=acd77(14)*acd77(13)
      acd77(54)=acd77(54)+acd77(56)+acd77(57)+acd77(55)+acd77(58)
      acd77(54)=acd77(54)*acd77(12)
      acd77(55)=acd77(22)*acd77(32)
      acd77(56)=acd77(20)*acd77(31)
      acd77(57)=acd77(18)*acd77(26)
      acd77(58)=acd77(16)*acd77(25)
      acd77(59)=acd77(14)*acd77(23)
      acd77(55)=acd77(59)+acd77(55)+acd77(56)+acd77(57)+acd77(58)
      acd77(56)=acd77(55)*acd77(24)
      acd77(57)=acd77(7)*acd77(6)
      acd77(58)=acd77(5)*acd77(4)
      acd77(59)=acd77(3)*acd77(2)
      acd77(57)=acd77(59)+acd77(57)+acd77(58)
      acd77(57)=acd77(57)*acd77(1)
      acd77(58)=acd77(7)*acd77(11)
      acd77(59)=acd77(5)*acd77(10)
      acd77(60)=acd77(3)*acd77(9)
      acd77(58)=acd77(60)+acd77(58)+acd77(59)
      acd77(59)=acd77(58)*acd77(8)
      acd77(60)=acd77(29)*acd77(4)
      acd77(61)=acd77(28)*acd77(2)
      acd77(60)=acd77(60)+acd77(61)
      acd77(60)=acd77(60)*acd77(27)
      acd77(61)=acd77(10)*acd77(29)
      acd77(62)=acd77(9)*acd77(28)
      acd77(61)=acd77(61)+acd77(62)
      acd77(61)=acd77(61)*acd77(30)
      acd77(54)=acd77(54)+acd77(57)+acd77(61)+acd77(59)+acd77(56)+acd77(60)+acd&
      &77(33)
      acd77(55)=acd77(40)*acd77(55)
      acd77(56)=acd77(22)*acd77(46)
      acd77(57)=acd77(20)*acd77(45)
      acd77(59)=acd77(18)*acd77(43)
      acd77(60)=acd77(16)*acd77(42)
      acd77(61)=acd77(14)*acd77(41)
      acd77(56)=acd77(47)+acd77(56)+acd77(60)+acd77(61)+acd77(57)+acd77(59)
      acd77(57)=acd77(12)*acd77(56)
      acd77(58)=acd77(34)*acd77(58)
      acd77(59)=acd77(29)*acd77(36)
      acd77(60)=acd77(28)*acd77(35)
      acd77(59)=acd77(60)+acd77(48)+acd77(59)
      acd77(59)=acd77(27)*acd77(59)
      acd77(60)=acd77(7)*acd77(37)
      acd77(61)=acd77(5)*acd77(36)
      acd77(60)=acd77(38)+acd77(60)+acd77(61)
      acd77(61)=acd77(3)*acd77(35)
      acd77(61)=acd77(61)+acd77(60)
      acd77(61)=acd77(1)*acd77(61)
      acd77(62)=acd77(33)*acd77(39)
      acd77(63)=acd77(11)*acd77(51)
      acd77(64)=acd77(29)*acd77(44)
      acd77(64)=acd77(64)+acd77(50)
      acd77(65)=acd77(10)*acd77(64)
      acd77(66)=acd77(28)*acd77(44)
      acd77(66)=acd77(66)+acd77(49)
      acd77(67)=acd77(9)*acd77(66)
      acd77(55)=acd77(57)+acd77(55)+acd77(61)+acd77(67)+acd77(65)+acd77(59)+2.0&
      &_ki*acd77(62)+acd77(63)+acd77(58)
      acd77(57)=ninjaP*acd77(54)
      acd77(56)=acd77(40)*acd77(56)
      acd77(58)=acd77(3)*acd77(34)
      acd77(58)=acd77(58)+acd77(66)
      acd77(58)=acd77(35)*acd77(58)
      acd77(59)=acd77(34)*acd77(60)
      acd77(60)=acd77(36)*acd77(64)
      acd77(61)=acd77(37)*acd77(51)
      acd77(62)=acd77(33)*acd77(52)
      acd77(63)=acd77(44)*acd77(48)
      acd77(56)=acd77(57)+acd77(56)+acd77(63)+acd77(62)+acd77(53)+acd77(61)+acd&
      &77(59)+acd77(58)+acd77(60)
      brack(ninjaidxt1mu0)=acd77(55)
      brack(ninjaidxt0mu0)=acd77(56)
      brack(ninjaidxt0mu2)=acd77(54)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d77h2_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd77h2_qp
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
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_ubaru_httbar_d77h2l131_qp
