module     p2_gg_httbar_d79h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d79h8l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd79h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc79(38)
      complex(ki) :: QspQ
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak2e1
      QspQ = dotproduct(Q,Q)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspk2 = dotproduct(Q,k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      acc79(1)=abb79(9)
      acc79(2)=abb79(10)
      acc79(3)=abb79(11)
      acc79(4)=abb79(12)
      acc79(5)=abb79(13)
      acc79(6)=abb79(14)
      acc79(7)=abb79(15)
      acc79(8)=abb79(16)
      acc79(9)=abb79(17)
      acc79(10)=abb79(18)
      acc79(11)=abb79(19)
      acc79(12)=abb79(20)
      acc79(13)=abb79(21)
      acc79(14)=abb79(22)
      acc79(15)=abb79(23)
      acc79(16)=abb79(24)
      acc79(17)=abb79(25)
      acc79(18)=abb79(26)
      acc79(19)=abb79(27)
      acc79(20)=abb79(28)
      acc79(21)=abb79(29)
      acc79(22)=abb79(30)
      acc79(23)=abb79(31)
      acc79(24)=abb79(32)
      acc79(25)=abb79(33)
      acc79(26)=abb79(34)
      acc79(27)=abb79(35)
      acc79(28)=abb79(39)
      acc79(29)=acc79(18)*QspQ
      acc79(30)=acc79(7)*Qspvae2e1
      acc79(31)=acc79(17)*Qspval4e1
      acc79(32)=acc79(20)*Qspval5e1
      acc79(33)=Qspvak2e2*acc79(25)
      acc79(34)=Qspvak2l5*acc79(10)
      acc79(35)=Qspk2*acc79(24)
      acc79(29)=acc79(35)+acc79(34)+acc79(33)+acc79(32)+acc79(31)+acc79(9)+acc7&
      &9(30)+acc79(29)
      acc79(29)=Qspvae1k2*acc79(29)
      acc79(30)=-acc79(2)*Qspval4e1
      acc79(31)=acc79(3)*Qspval5e1
      acc79(32)=acc79(4)*Qspvae2e1
      acc79(33)=-acc79(19)*Qspvae1e2
      acc79(34)=acc79(28)*Qspvae1l5
      acc79(30)=acc79(6)+acc79(34)+acc79(33)+acc79(32)+acc79(30)+acc79(31)
      acc79(30)=QspQ*acc79(30)
      acc79(31)=-Qspval3e2*acc79(19)
      acc79(32)=Qspval3l5*acc79(28)
      acc79(33)=Qspval3k2*acc79(18)
      acc79(31)=acc79(33)+acc79(32)+acc79(31)+acc79(11)
      acc79(31)=Qspvae1l3*acc79(31)
      acc79(32)=Qspvae2l3*acc79(4)
      acc79(33)=Qspval5l3*acc79(3)
      acc79(34)=-Qspval4l3*acc79(2)
      acc79(32)=acc79(34)+acc79(33)+acc79(32)+acc79(16)
      acc79(32)=Qspval3e1*acc79(32)
      acc79(33)=Qspvae2k2*acc79(15)
      acc79(34)=Qspval5k2*acc79(26)
      acc79(35)=Qspval4k2*acc79(27)
      acc79(33)=acc79(35)+acc79(34)+acc79(33)+acc79(5)
      acc79(33)=Qspvak2e1*acc79(33)
      acc79(34)=acc79(22)*Qspvae1e2
      acc79(35)=acc79(23)*Qspvae1l5
      acc79(34)=acc79(35)+acc79(34)+acc79(21)
      acc79(34)=Qspval4e1*acc79(34)
      acc79(35)=acc79(8)*Qspvae2e1
      acc79(36)=acc79(12)*Qspvae1e2
      acc79(37)=acc79(13)*Qspvae1l5
      acc79(38)=acc79(14)*Qspval5e1
      brack=acc79(1)+acc79(29)+acc79(30)+acc79(31)+acc79(32)+acc79(33)+acc79(34&
      &)+acc79(35)+acc79(36)+acc79(37)+acc79(38)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d79h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd79h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d79
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d79 = 0.0_ki
      d79 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d79, ki), aimag(d79), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d79h8l1
