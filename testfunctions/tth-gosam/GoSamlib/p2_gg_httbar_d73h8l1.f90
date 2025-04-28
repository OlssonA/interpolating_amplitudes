module     p2_gg_httbar_d73h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d73h8l1.f90
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
      use p2_gg_httbar_abbrevd73h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc73(48)
      complex(ki) :: QspQ
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak2e2
      QspQ = dotproduct(Q,Q)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspk2 = dotproduct(Q,k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      acc73(1)=abb73(9)
      acc73(2)=abb73(10)
      acc73(3)=abb73(11)
      acc73(4)=abb73(12)
      acc73(5)=abb73(13)
      acc73(6)=abb73(14)
      acc73(7)=abb73(15)
      acc73(8)=abb73(16)
      acc73(9)=abb73(17)
      acc73(10)=abb73(18)
      acc73(11)=abb73(19)
      acc73(12)=abb73(20)
      acc73(13)=abb73(21)
      acc73(14)=abb73(22)
      acc73(15)=abb73(23)
      acc73(16)=abb73(24)
      acc73(17)=abb73(25)
      acc73(18)=abb73(26)
      acc73(19)=abb73(27)
      acc73(20)=abb73(28)
      acc73(21)=abb73(29)
      acc73(22)=abb73(30)
      acc73(23)=abb73(31)
      acc73(24)=abb73(32)
      acc73(25)=abb73(33)
      acc73(26)=abb73(34)
      acc73(27)=abb73(35)
      acc73(28)=abb73(36)
      acc73(29)=abb73(37)
      acc73(30)=abb73(38)
      acc73(31)=abb73(39)
      acc73(32)=abb73(40)
      acc73(33)=abb73(41)
      acc73(34)=abb73(42)
      acc73(35)=abb73(43)
      acc73(36)=abb73(44)
      acc73(37)=-acc73(12)*QspQ
      acc73(38)=acc73(7)*Qspval5e2
      acc73(39)=acc73(21)*Qspvak1e2
      acc73(40)=acc73(22)*Qspvae1e2
      acc73(41)=-acc73(26)*Qspval4e2
      acc73(42)=Qspvak2e1*acc73(28)
      acc73(43)=Qspvak2l5*acc73(35)
      acc73(44)=Qspvak2k1*acc73(36)
      acc73(45)=Qspk2*acc73(18)
      acc73(37)=acc73(45)+acc73(44)+acc73(43)+acc73(42)+acc73(41)+acc73(40)+acc&
      &73(39)+acc73(16)+acc73(38)+acc73(37)
      acc73(37)=Qspvae2k2*acc73(37)
      acc73(38)=acc73(3)*Qspval4e2
      acc73(39)=-acc73(13)*Qspvae2e1
      acc73(40)=acc73(14)*Qspval5e2
      acc73(41)=-acc73(19)*Qspvak1e2
      acc73(42)=-acc73(24)*Qspvae2k1
      acc73(43)=acc73(30)*Qspvae2l5
      acc73(44)=acc73(32)*Qspvae1e2
      acc73(38)=acc73(4)+acc73(44)+acc73(43)+acc73(42)+acc73(41)+acc73(40)+acc7&
      &3(39)+acc73(38)
      acc73(38)=QspQ*acc73(38)
      acc73(39)=-Qspval3e1*acc73(13)
      acc73(40)=Qspval3l5*acc73(30)
      acc73(41)=-Qspval3k2*acc73(12)
      acc73(42)=-Qspval3k1*acc73(24)
      acc73(39)=acc73(42)+acc73(41)+acc73(40)+acc73(39)+acc73(6)
      acc73(39)=Qspvae2l3*acc73(39)
      acc73(40)=Qspvae1l3*acc73(32)
      acc73(41)=Qspval5l3*acc73(14)
      acc73(42)=Qspval4l3*acc73(3)
      acc73(43)=-Qspvak1l3*acc73(19)
      acc73(40)=acc73(43)+acc73(42)+acc73(41)+acc73(40)+acc73(2)
      acc73(40)=Qspval3e2*acc73(40)
      acc73(41)=Qspvae1k2*acc73(23)
      acc73(42)=Qspval5k2*acc73(31)
      acc73(43)=Qspval4k2*acc73(33)
      acc73(44)=Qspvak1k2*acc73(34)
      acc73(41)=acc73(44)+acc73(43)+acc73(42)+acc73(41)+acc73(25)
      acc73(41)=Qspvak2e2*acc73(41)
      acc73(42)=acc73(9)*Qspvae2e1
      acc73(43)=acc73(11)*Qspvae2l5
      acc73(44)=acc73(15)*Qspvae2k1
      acc73(42)=acc73(44)+acc73(43)+acc73(42)+acc73(5)
      acc73(42)=Qspval4e2*acc73(42)
      acc73(43)=acc73(8)*Qspvae2e1
      acc73(44)=acc73(10)*Qspvae1e2
      acc73(45)=acc73(17)*Qspval5e2
      acc73(46)=acc73(20)*Qspvak1e2
      acc73(47)=acc73(27)*Qspvae2k1
      acc73(48)=acc73(29)*Qspvae2l5
      brack=acc73(1)+acc73(37)+acc73(38)+acc73(39)+acc73(40)+acc73(41)+acc73(42&
      &)+acc73(43)+acc73(44)+acc73(45)+acc73(46)+acc73(47)+acc73(48)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d73h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd73h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d73
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d73 = 0.0_ki
      d73 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d73, ki), aimag(d73), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d73h8l1
