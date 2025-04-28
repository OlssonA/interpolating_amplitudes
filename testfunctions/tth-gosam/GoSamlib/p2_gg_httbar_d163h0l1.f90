module     p2_gg_httbar_d163h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d163h0l1.f90
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
      use p2_gg_httbar_abbrevd163h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc163(60)
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspl5
      complex(ki) :: Qspk2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: QspQ
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspl5 = dotproduct(Q,l5)
      Qspk2 = dotproduct(Q,k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      QspQ = dotproduct(Q,Q)
      acc163(1)=abb163(11)
      acc163(2)=abb163(12)
      acc163(3)=abb163(13)
      acc163(4)=abb163(14)
      acc163(5)=abb163(15)
      acc163(6)=abb163(16)
      acc163(7)=abb163(17)
      acc163(8)=abb163(18)
      acc163(9)=abb163(19)
      acc163(10)=abb163(20)
      acc163(11)=abb163(21)
      acc163(12)=abb163(22)
      acc163(13)=abb163(23)
      acc163(14)=abb163(24)
      acc163(15)=abb163(25)
      acc163(16)=abb163(26)
      acc163(17)=abb163(27)
      acc163(18)=abb163(28)
      acc163(19)=abb163(29)
      acc163(20)=abb163(30)
      acc163(21)=abb163(31)
      acc163(22)=abb163(33)
      acc163(23)=abb163(45)
      acc163(24)=abb163(46)
      acc163(25)=abb163(48)
      acc163(26)=abb163(50)
      acc163(27)=abb163(52)
      acc163(28)=abb163(54)
      acc163(29)=abb163(56)
      acc163(30)=abb163(62)
      acc163(31)=abb163(80)
      acc163(32)=abb163(83)
      acc163(33)=abb163(84)
      acc163(34)=abb163(154)
      acc163(35)=abb163(166)
      acc163(36)=acc163(13)*Qspval5k2
      acc163(37)=acc163(16)*Qspval5k1
      acc163(38)=acc163(17)*Qspval5l4
      acc163(39)=acc163(23)*Qspl5
      acc163(40)=acc163(30)*Qspk2
      acc163(41)=acc163(31)*Qspval5e2
      acc163(42)=Qspvae2k2*acc163(29)
      acc163(43)=Qspval4k2*acc163(20)
      acc163(44)=Qspvak1k2*acc163(7)
      acc163(36)=acc163(44)+acc163(43)+acc163(42)+acc163(41)+acc163(40)+acc163(&
      &39)+acc163(38)+acc163(37)+acc163(36)+acc163(5)
      acc163(36)=Qspe1*acc163(36)
      acc163(37)=acc163(8)*Qspl5
      acc163(38)=acc163(10)*Qspval5k2
      acc163(39)=acc163(15)*Qspval5k1
      acc163(40)=acc163(19)*Qspval5l4
      acc163(41)=acc163(26)*Qspk2
      acc163(42)=acc163(28)*Qspval5e2
      acc163(43)=Qspvae2e1*acc163(27)
      acc163(44)=Qspvae1e2*acc163(4)
      acc163(45)=Qspvae2l5*acc163(21)
      acc163(46)=Qspvae1l5*acc163(33)
      acc163(47)=Qspval5e1*acc163(22)
      acc163(48)=-Qspvae1l4*acc163(34)
      acc163(49)=Qspval4e1*acc163(24)
      acc163(50)=Qspvak2e2*acc163(2)
      acc163(51)=Qspvae1k2*acc163(6)
      acc163(52)=Qspvak2e1*acc163(25)
      acc163(53)=Qspvae1k1*acc163(18)
      acc163(54)=Qspvak1e1*acc163(12)
      acc163(55)=Qspval4l5*acc163(14)
      acc163(56)=Qspvak2l5*acc163(1)
      acc163(57)=Qspvak2l4*acc163(35)
      acc163(58)=Qspvak2k1*acc163(9)
      acc163(59)=Qspvak1l5*acc163(11)
      acc163(60)=QspQ*acc163(32)
      brack=acc163(3)+acc163(36)+acc163(37)+acc163(38)+acc163(39)+acc163(40)+ac&
      &c163(41)+acc163(42)+acc163(43)+acc163(44)+acc163(45)+acc163(46)+acc163(4&
      &7)+acc163(48)+acc163(49)+acc163(50)+acc163(51)+acc163(52)+acc163(53)+acc&
      &163(54)+acc163(55)+acc163(56)+acc163(57)+acc163(58)+acc163(59)+acc163(60)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d163h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd163h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d163
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k4-k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d163 = 0.0_ki
      d163 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d163, ki), aimag(d163), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d163h0l1
